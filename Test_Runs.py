import argparse
import warnings

# Standard imports
import numpy as np
import pandas as pd
import torch

# For Gower distance
import gower

# For data preprocessing
from rdt import HyperTransformer
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For the SUPPORT dataset
from pycox.datasets import support

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

# SDV aspects
from sdv.evaluation import evaluate

from sdv.metrics.tabular import NumericalLR, NumericalMLP, NumericalSVR

# Load in the support data
data_supp = support.read_df()

###############################################################################
# DATA PREPROCESSING #
# We one-hot the categorical cols and standardise the continuous cols
data_supp["x14"] = data_supp["x0"]
# data_supp = data_supp.astype('float32')
data_supp = data_supp[
    ["duration"] + [f"x{i}" for i in range(1, 15)] + ["event"]
]
data_supp[["x1", "x2", "x3", "x4", "x5", "x6", "event"]] = data_supp[
    ["x1", "x2", "x3", "x4", "x5", "x6", "event"]
].astype(int)
transformer_dtypes = {
    "i": "one_hot_encoding",
    "f": "numerical",
    "O": "one_hot_encoding",
    "b": "one_hot_encoding",
    "M": "datetime",
}
ht = HyperTransformer(dtype_transformers=transformer_dtypes)
ht.fit(data_supp)
transformed = ht.transform(data_supp)
cat_cols = [f"x{i}" for i in range(1, 7)] + ["event"]
cont_cols = [f"x{i}" for i in range(7, 15)] + ["duration"]
num_categories = (
    np.array([np.amax(data_supp[col]) for col in cat_cols]) + 1
).astype(int)
num_continuous = len(cont_cols)
cols_standardize = transformed.columns[:num_continuous]
cols_leave = transformed.columns[num_continuous:]
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [([col], None) for col in cols_leave]
x_mapper = DataFrameMapper(leave + standardize)
x_train_df = x_mapper.fit_transform(transformed)
x_train_df = x_mapper.transform(transformed)
x_train = x_train_df.astype("float32")
###############################################################################
# Prepare data for interaction with torch VAE
Y = torch.Tensor(x_train)
dataset = TensorDataset(Y)
batch_size = 32

generator = None
sample_rate = batch_size / len(dataset)
data_loader = DataLoader(
    dataset,
    batch_sampler=UniformWithReplacementSampler(
        num_samples=len(dataset), sample_rate=sample_rate, generator=generator
    ),
    pin_memory=True,
    generator=generator,
)

# shuffle = True
# data_loader = DataLoader(
#     dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle
# )

# Define lists to contain the metrics achieved on the
# train/generate/evaluate runs
bns = []
lrs = []
svcs = []
gmlls = []
cs = []
ks = []
kses = []
contkls = []
disckls = []
lr_privs = []
mlp_privs = []
svr_privs = []
gowers = []

target_delta = 1e-3
target_eps = 10.0

# Create VAE
latent_dim = 2
encoder = Encoder(x_train.shape[1], latent_dim)
decoder = Decoder(
    latent_dim, num_continuous, num_categories=num_categories
)
vae = VAE(encoder, decoder)

vae.train(data_loader, n_epochs=5)