#%% -------- Import Libraries -------- #

# Standard imports
import numpy as np
import pandas as pd
import torch

# VAE is in other folder
import sys

sys.path.append("../")

from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For the SUPPORT dataset
from pycox.datasets import support

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

from utils import (
    set_seed,
    support_pre_proc,
    plot_elbo,
    plot_likelihood_breakdown,
    plot_variable_distributions,
    reverse_transformers,
)
from metrics import distribution_metrics, privacy_metrics

import warnings

warnings.filterwarnings(
    "ignore"
)  # We suppress warnings to avoid SDMETRICS throwing unique synthetic data warnings (i.e.
# data in synthetic set is not in the real data set) as well as SKLEARN throwing convergence warnings (pre-processing uses
# GMM from sklearn and this throws non convergence warnings)

set_seed(0)

# Load in the support data
data_supp = support.read_df()

# Column Definitions
original_continuous_columns = ["duration"] + [f"x{i}" for i in range(7, 15)]
original_categorical_columns = ["event"] + [f"x{i}" for i in range(1, 7)]
#%% -------- Data Pre-Processing -------- #

pre_proc_method = "standard"

(
    x_train,
    data_supp,
    reordered_dataframe_columns,
    continuous_transformers,
    categorical_transformers,
    num_categories,
    num_continuous,
) = support_pre_proc(data_supp=data_supp, pre_proc_method=pre_proc_method)
#%% -------- Create & Train VAE -------- #

# User defined hyperparams
# General training
batch_size = 32
latent_dim = 8
hidden_dim = 32
n_epochs = 5
logging_freq = 1  # Number of epochs we should log the results to the user
patience = 5  # How many epochs should we allow the model train to see if
# improvement is made
delta = 10  # The difference between elbo values that registers an improvement
filepath = None  # Where to save the best model


# Privacy params
differential_privacy = False  # Do we want to implement differential privacy
sample_rate = 0.1  # Sampling rate
C = 1e16  # Clipping threshold - any gradients above this are clipped
noise_scale = None  # Noise multiplier - influences how much noise to add
target_eps = 1  # Target epsilon for privacy accountant
target_delta = 1e-5  # Target delta for privacy accountant

# Prepare data for interaction with torch VAE
Y = torch.Tensor(x_train)
dataset = TensorDataset(Y)

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
decoder = Decoder(latent_dim, num_continuous, num_categories=num_categories)

vae = VAE(encoder, decoder)

print(vae)

if differential_privacy == False:
    (
        training_epochs,
        log_elbo,
        log_reconstruction,
        log_divergence,
        log_categorical,
        log_numerical,
    ) = vae.train(
        data_loader,
        n_epochs=n_epochs,
        logging_freq=logging_freq,
        patience=patience,
        delta=delta,
    )

elif differential_privacy == True:
    (
        training_epochs,
        log_elbo,
        log_reconstruction,
        log_divergence,
        log_categorical,
        log_numerical,
    ) = vae.diff_priv_train(
        data_loader,
        n_epochs=n_epochs,
        logging_freq=logging_freq,
        patience=patience,
        delta=delta,
        C=C,
        target_eps=target_eps,
        target_delta=target_delta,
        sample_rate=sample_rate,
        noise_scale=noise_scale,
    )
    print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")
#%% -------- Plot Loss Features ELBO Breakdown -------- #

elbo_fig = plot_elbo(
    n_epochs=training_epochs,
    log_elbo=log_elbo,
    log_reconstruction=log_reconstruction,
    log_divergence=log_divergence,
    saving_filepath="",
    pre_proc_method=pre_proc_method,
)
#%% -------- Plot Loss Features Reconstruction Breakdown -------- #

likelihood_fig = plot_likelihood_breakdown(
    n_epochs=training_epochs,
    log_categorical=log_categorical,
    log_numerical=log_numerical,
    saving_filepath="",
    pre_proc_method=pre_proc_method,
)
#%% -------- Synthetic Data Generation -------- #

synthetic_sample = vae.generate(data_supp.shape[0])

synthetic_sample = pd.DataFrame(
    synthetic_sample.cpu().detach().numpy(),
    columns=reordered_dataframe_columns,
)

# Reverse the transformations

synthetic_supp = reverse_transformers(
    synthetic_set=synthetic_sample,
    data_supp_columns=data_supp.columns,
    cont_transformers=continuous_transformers,
    cat_transformers=categorical_transformers,
    pre_proc_method=pre_proc_method,
)

#%% -------- Plot Histograms For All The Variable Distributions -------- #

plot_variable_distributions(
    categorical_columns=original_categorical_columns,
    continuous_columns=original_continuous_columns,
    data_supp=data_supp,
    synthetic_supp=synthetic_supp,
    saving_filepath="",
    pre_proc_method=pre_proc_method,
)
#%% -------- SDV Metrics -------- #

# Define the metrics you want the model to evaluate

gower = False

# Define distributional metrics required - for sdv_baselines this is set by default
distributional_metrics = [
    "SVCDetection",
    "GMLogLikelihood",
    "CSTest",
    "KSTest",
    "KSTestExtended",
    "ContinuousKLDivergence",
    "DiscreteKLDivergence",
]

metrics = distribution_metrics(
    gower_bool=gower,
    distributional_metrics=distributional_metrics,
    data_supp=data_supp,
    synthetic_supp=synthetic_supp,
    categorical_columns=original_categorical_columns,
    continuous_columns=original_continuous_columns,
    saving_filepath="",
    pre_proc_method=pre_proc_method,
)

#%% -------- Privacy Metrics -------- #

# Specify our private variable

private_variable = "x14"

privacy_metric = privacy_metrics(
    private_variable=private_variable,
    data_supp=data_supp,
    synthetic_supp=synthetic_supp,
    categorical_columns=original_categorical_columns,
    continuous_columns=original_continuous_columns,
    saving_filepath=None,
    pre_proc_method=pre_proc_method,
)
