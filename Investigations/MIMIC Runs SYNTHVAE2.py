#%% -------- Import Libraries -------- #

# Standard imports
from tokenize import String
import numpy as np
import pandas as pd
import torch

# For Gower distance
import gower

# VAE is in other folder
import sys
sys.path.append('../')

from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

# SDV aspects
from sdv.evaluation import evaluate

from sdv.metrics.tabular import NumericalLR, NumericalMLP, NumericalSVR

from rdt.transformers import categorical, numerical, datetime

# Graph Visualisation
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import mimic_pre_proc, constraint_filtering, plot_elbo, plot_likelihood_breakdown, plot_variable_distributions, metric_calculation

# Load in the mimic single table data 

filepath = "C:/Users/dxb085/Documents/NHSX Internship/Private MIMIC Data/table_one_synthvae.csv"

data_supp = pd.read_csv(filepath)
# Save the original columns

original_categorical_columns = ['ETHNICITY', 'DISCHARGE_LOCATION', 'GENDER', 'FIRST_CAREUNIT', 'VALUEUOM', 'LABEL']
original_continuous_columns = ['Unnamed: 0', 'ROW_ID', 'SUBJECT_ID', 'VALUE', 'age']
original_datetime_columns = ['ADMITTIME', 'DISCHTIME', 'DOB', 'CHARTTIME']

# Drop DOD column as it contains NANS - for now

#data_supp = data_supp.drop('DOD', axis = 1)

original_columns = original_categorical_columns + original_continuous_columns + original_datetime_columns
#%% -------- Data Pre-Processing -------- #

x_train, original_metric_set, reordered_dataframe_columns, continuous_transformers, categorical_transformers, datetime_transformers, num_categories, num_continuous = mimic_pre_proc(data_supp=data_supp)

#%% -------- Create & Train VAE -------- #

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
latent_dim = 256
hidden_dim = 256
encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
decoder = Decoder(
    latent_dim, num_continuous, num_categories=num_categories
)

vae = VAE(encoder, decoder)

n_epochs = 5

log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.train(data_loader, n_epochs=n_epochs)

#%% -------- Plot Loss Features ELBO Breakdown -------- #

plot_elbo(
    n_epochs=n_epochs, log_elbo=log_elbo, log_reconstruction=log_reconstruction,
    log_divergence=log_divergence, saving_filepath="", pre_proc_method="GMM"
)
#%% -------- Plot Loss Features Reconstruction Breakdown -------- #

plot_likelihood_breakdown(
    n_epochs=n_epochs, log_categorical=log_categorical, log_numerical=log_numerical,
    saving_filepath="", pre_proc_method="GMM"
)

#%% -------- Constraint Sampling -------- #

synthetic_supp = constraint_filtering(
    n_rows=data_supp.shape[0], vae=vae, reordered_cols=reordered_dataframe_columns,
    data_supp_columns=data_supp.columns, cont_transformers=continuous_transformers,
    cat_transformers=categorical_transformers, date_transformers=datetime_transformers
)
#%% -------- Plot Histograms For All The Variable Distributions -------- #

plot_variable_distributions(
    categorical_columns=original_categorical_columns, continuous_columns=original_continuous_columns,
    data_supp=data_supp, synthetic_supp=synthetic_supp,saving_filepath="",
    pre_proc_method="GMM"
)
#%% -------- SDV Metrics -------- #

# Define the metrics you want the model to evaluate

user_metrics = ['ContinuousKLDivergence', 'DiscreteKLDivergence']

metrics = metric_calculation(
    user_metrics=user_metrics, data_supp=data_supp, synthetic_supp=synthetic_supp,
    categorical_columns=original_categorical_columns, continuous_columns=original_continuous_columns,
    saving_filepath="", pre_proc_method="GMM"
)