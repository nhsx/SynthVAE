#%%
import argparse
from email.policy import default
import warnings

# Standard imports
import numpy as np
import pandas as pd
import torch

# For Gower distance
import gower

# For data preprocessing
from rdt import HyperTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

#torch.cuda.is_available()
#torch.cuda.current_device()
#torch.cuda.get_device_name(0)

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

# As of coding this, new version of RDT adds in GMM transformer which is what we require, however hyper transformers do not work as individual
# transformers take a 'columns' argument that can only allow for fitting of one column - so you need to loop over and create one for each column
# in order to fit the dataset - https://github.com/sdv-dev/RDT/issues/376

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

# Create VAE
latent_dim = 2
hidden_dim = 256
encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
decoder = Decoder(
    latent_dim, num_continuous, num_categories=num_categories
)
vae = VAE(encoder, decoder)

n_epochs = 200

log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.train(data_loader, n_epochs=n_epochs)

#%% -------- Plotting features for loss -------- #

import matplotlib.pyplot as plt

# Plot and save the breakdown of ELBO

x1 = np.arange(n_epochs)
y1 = log_elbo
# plotting the elbo
plt.plot(x1, y1, label = "ELBO")
# line 2 points
y2 = log_reconstruction
# plotting the reconstruction term
plt.plot(x1, y2, label = "Reconstruction Term")
# plotting the divergence term
plt.plot(x1, log_divergence, label = "Divergence Term")
plt.xlabel('Number of Epochs')
# Set the y axis label of the current axis.
plt.ylabel('Loss Value')
# Set a title of the current axes.
plt.title('Breakdown of the ELBO - 256 Latent Dim')
# show a legend on the plot
plt.legend()
#plt.savefig("Elbo Breakdown at 256 Latent Dim.png")
# Display a figure.
plt.show()

# Plot and save the breakdown of the reconstruction term

# Clear plot after saving
plt.clf()

x1 = np.arange(n_epochs)
y1 = log_categorical
# plotting the elbo
plt.subplot(1,2,1)
plt.plot(x1, y1, label = "Categorical Reconstruction")
# line 2 points
y2 = log_numerical
# plotting the reconstruction term
plt.subplot(1,2,2)
plt.plot(x1, y2, label = "Numerical Reconstruction")
plt.xlabel('Number of Epochs')
# Set the y axis label of the current axis.
plt.ylabel('Loss Value')
# Set a title of the current axes.
plt.title('Breakdown of the Reconstruction Term - 256 Latent Dim')
# show a legend on the plot
plt.legend()
plt.tight_layout()
#plt.savefig("Reconstruction Breakdown at 256 Latent Dim.png")
# Display a figure.
plt.show()
#%% -------- Plotting features for synthetic data distribution -------- #

# Generate a synthetic set using trained vae

synthetic_trial = vae.generate(8873) # 8873 is size of support

#%%

print(synthetic_trial[:,1].detach().numpy())
#%%
# Now choose columns you want to do histograms for (for sake of brevity) and compare to support visually

cat_columns = [1, 4]
cont_columns = [28, 31]

for column in cat_columns:
    # Plot these cat_columns against the original columns in x_train
    plt.subplot(1,2,1)
    plt.hist(synthetic_trial[:,column].detach().numpy())
    plt.xlabel("Value")
    plt.ylabel("Counts")
    plt.title("Synthetic Arm - Categorical {}".format(str(column)))
    plt.subplot(1,2,2)
    plt.hist(x_train[:,column])
    plt.xlabel("Value")
    plt.ylabel("Counts")
    plt.title("Original Arm - Categorical {}".format(str(column)))
    plt.tight_layout()
    plt.show()

for column in cont_columns:
    # Plot these cont_columns against the original columns in x_train
    plt.subplot(1,2,1)
    plt.hist(synthetic_trial[:,column].detach().numpy())
    plt.xlabel("Value")
    plt.ylabel("Counts")
    plt.title("Synthetic Arm - Continuous {}".format(str(column)))
    plt.subplot(1,2,2)
    plt.hist(x_train[:,column])
    plt.xlabel("Value")
    plt.ylabel("Counts")
    plt.title("Original Arm - Continuous {}".format(str(column)))
    plt.tight_layout()
    plt.show() 