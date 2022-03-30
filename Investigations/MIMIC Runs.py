#%% -------- Import Libraries -------- #

# Standard imports
from tokenize import String
import numpy as np
import pandas as pd
import torch

# VAE is in other folder
import sys
sys.path.append('../')

from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

from rdt.transformers import datetime

from utils import mimic_pre_proc, constraint_filtering, plot_elbo, plot_likelihood_breakdown, plot_variable_distributions
from metrics import distribution_metrics, privacy_metrics

# Load in the mimic single table data 

filepath = ".../Private MIMIC Data/table_one_synthvae.csv"

data_supp = pd.read_csv(filepath)
# Save the original columns

original_categorical_columns = ['ETHNICITY', 'DISCHARGE_LOCATION', 'GENDER', 'FIRST_CAREUNIT', 'VALUEUOM', 'LABEL']
original_continuous_columns = ['Unnamed: 0', 'ROW_ID', 'SUBJECT_ID', 'VALUE', 'age']
original_datetime_columns = ['ADMITTIME', 'DISCHTIME', 'DOB', 'CHARTTIME']

# Drop DOD column as it contains NANS - for now

#data_supp = data_supp.drop('DOD', axis = 1)

original_columns = original_categorical_columns + original_continuous_columns + original_datetime_columns

#%% -------- Data Pre-Processing -------- #

pre_proc_method = "standard"
x_train, original_metric_set, reordered_dataframe_columns, continuous_transformers, categorical_transformers, datetime_transformers, num_categories, num_continuous = mimic_pre_proc(data_supp=data_supp, pre_proc_method=pre_proc_method)

#%% -------- Create & Train VAE -------- #

# User defined hyperparams
# General training
batch_size=32
latent_dim=256
hidden_dim=256
n_epochs=5
logging_freq=1 # Number of epochs we should log the results to the user
patience=5 # How many epochs should we allow the model train to see if
# improvement is made
delta=10 # The difference between elbo values that registers an improvement
filepath=None # Where to save the best model


# Privacy params
differential_privacy = False # Do we want to implement differential privacy
sample_rate=0.1 # Sampling rate
C = 1e16 # Clipping threshold - any gradients above this are clipped
noise_scale=None # Noise multiplier - influences how much noise to add
target_eps=1 # Target epsilon for privacy accountant
target_delta=1e-5 # Target delta for privacy accountant

# Prepare data for interaction with torch VAE
Y = torch.Tensor(x_train)
dataset = TensorDataset(Y)

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
encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
decoder = Decoder(
    latent_dim, num_continuous, num_categories=num_categories
)

vae = VAE(encoder, decoder)

if(differential_privacy==False):
    training_epochs, log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.train(data_loader, n_epochs=n_epochs)
    
elif(differential_privacy==True):
    training_epochs, log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.diff_priv_train(
            data_loader,
            n_epochs=n_epochs,
            C=C,
            target_eps=target_eps,
            target_delta=target_delta,
            sample_rate=sample_rate,
            noise_scale=noise_scale
        )
    print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")
#%% -------- Plot Loss Features ELBO Breakdown -------- #

elbo_fig = plot_elbo(
    n_epochs=training_epochs, log_elbo=log_elbo, log_reconstruction=log_reconstruction,
    log_divergence=log_divergence, saving_filepath="", pre_proc_method=pre_proc_method
)
#%% -------- Plot Loss Features Reconstruction Breakdown -------- #

likelihood_fig = plot_likelihood_breakdown(
    n_epochs=training_epochs, log_categorical=log_categorical, log_numerical=log_numerical,
    saving_filepath="", pre_proc_method=pre_proc_method
)
#%% -------- Constraint Sampling -------- #

synthetic_supp = constraint_filtering(
    n_rows=data_supp.shape[0], vae=vae, reordered_cols=reordered_dataframe_columns,
    data_supp_columns=data_supp.columns, cont_transformers=continuous_transformers,
    cat_transformers=categorical_transformers, date_transformers=datetime_transformers,
    pre_proc_method=pre_proc_method
)
#%% -------- Plot Histograms For All The Variable Distributions -------- #

plot_variable_distributions(
    categorical_columns=original_categorical_columns, continuous_columns=original_continuous_columns,
    data_supp=data_supp, synthetic_supp=synthetic_supp,saving_filepath="",
    pre_proc_method=pre_proc_method
)
#%% -------- Datetime Handling -------- #

# If the dataset has datetimes then we need to re-convert these to a numerical
# Value representing seconds, this is so we can evaluate the metrics on them

metric_synthetic_supp = synthetic_supp.copy()

for index, column in enumerate(original_datetime_columns):

        # Fit datetime transformer - converts to seconds
        temp_datetime = datetime.DatetimeTransformer()
        temp_datetime.fit(metric_synthetic_supp, columns = column)

        metric_synthetic_supp = temp_datetime.transform(metric_synthetic_supp)

#%% -------- SDV Metrics -------- #

# Define the metrics you want the model to evaluate

gower=False

metrics = distribution_metrics(
    gower=gower, data_supp=original_metric_set, synthetic_supp=metric_synthetic_supp,
    categorical_columns=original_categorical_columns, continuous_columns=original_continuous_columns,
    saving_filepath="", pre_proc_method=pre_proc_method
)

#%% -------- Privacy Metrics -------- #

# Specify our private variable

private_variable = 'x14'

privacy_metric = privacy_metrics(private_variable=private_variable, data_supp=data_supp,
                                synthetic_supp=synthetic_supp, categorical_columns=original_categorical_columns,
                                continuous_columns=original_continuous_columns, saving_filepath=None, pre_proc_method=pre_proc_method)