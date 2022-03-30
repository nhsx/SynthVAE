#%% -------- Import Libraries -------- #

# Standard imports
from webbrowser import GenericBrowser
import numpy as np
import pandas as pd
import torch

# VAE is in other folder as well as opacus adapted library
import sys
sys.path.append('../')

# Opacus support for differential privacy
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For the SUPPORT dataset
from pycox.datasets import support

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

# Utility file contains all functions required to run notebook
from utils import support_pre_proc, plot_elbo, plot_likelihood_breakdown, plot_variable_distributions, reverse_transformers
from metrics import distribution_metrics

import optuna
import pickle

# Load in the support data
data_supp = support.read_df()

# Save the original columns

original_continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
original_categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 

original_columns = original_categorical_columns + original_continuous_columns
#%% -------- Data Pre-Processing -------- #

pre_proc_method = "GMM"

x_train, data_supp, reordered_dataframe_columns, continuous_transformers, categorical_transformers, num_categories, num_continuous = support_pre_proc(data_supp=data_supp, pre_proc_method=pre_proc_method)

#%% -------- Create & Train VAE -------- #

# User defined parameters

# General training
batch_size=32
n_epochs=5
logging_freq=1 # Number of epochs we should log the results to the user
patience=5 # How many epochs should we allow the model train to see if
# improvement is made
delta=10 # The difference between elbo values that registers an improvement
filepath=None # Where to save the best model


# Privacy params
differential_privacy = False # Do we want to implement differential privacy
sample_rate=0.1 # Sampling rate
noise_scale=None # Noise multiplier - influences how much noise to add
target_eps=1 # Target epsilon for privacy accountant
target_delta=1e-5 # Target delta for privacy accountant

# Define the metrics you want the model to evaluate

gower=False

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

# -------- Define our Optuna trial -------- #

def objective(trial, user_metrics, differential_privacy=False, target_delta=1e-3, target_eps=10.0, n_epochs=50):

    latent_dim = trial.suggest_int('Latent Dimension', 2, 128, step=2) # Hyperparam
    hidden_dim = trial.suggest_int('Hidden Dimension', 32, 1024, step=32) # Hyperparam

    encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
    decoder = Decoder(
        latent_dim, num_continuous, num_categories=num_categories
    )

    lr = trial.suggest_float('Learning Rate', 1e-3, 1e-2, step=1e-5)
    vae = VAE(encoder, decoder, lr=1e-3) # lr hyperparam

    C = trial.suggest_int('C', 10, 1e4, step=50) # Clipping hyperparam

    if differential_privacy == True:
        training_epochs, log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.diff_priv_train(
            data_loader,
            n_epochs=n_epochs,
            C=C, # Hyperparam
            target_eps=target_eps,
            target_delta=target_delta, 
            sample_rate=sample_rate,
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")

    else:

        training_epochs, log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.train(data_loader, n_epochs=n_epochs)

    # -------- Synthetic Data Generation -------- #

    synthetic_sample = vae.generate(data_supp.shape[0])

    if(torch.cuda.is_available()):
        synthetic_sample = pd.DataFrame(synthetic_sample.cpu().detach(), columns=reordered_dataframe_columns)
    else:
        synthetic_sample = pd.DataFrame(synthetic_sample.detach(), columns=reordered_dataframe_columns)

    # Reverse the transformations

    synthetic_supp = reverse_transformers(synthetic_set=synthetic_sample, data_supp_columns=data_supp.columns, 
                                      cont_transformers=continuous_transformers, cat_transformers=categorical_transformers,
                                      pre_proc_method=pre_proc_method
                                     )
    # -------- SDV Metrics -------- #

    metrics = distribution_metrics(
        gower=gower, data_supp=data_supp, synthetic_supp=synthetic_supp,
        categorical_columns=original_categorical_columns, continuous_columns=original_continuous_columns,
        saving_filepath=None, pre_proc_method=pre_proc_method
    )

    # Optuna wants a list of values in float form

    list_metrics = [metrics[i] for i in metrics.columns]

    return list_metrics

#%% -------- Run Hyperparam Optimisation -------- #

# If there is no study object in your folder then run and save the study so
# It can be resumed if needed

first_run=True  # First run indicates if we are creating a new hyperparam study

if(first_run==True):

    if(gower==True):
        directions = ['maximize' for i in range(8)]
    else:
        directions =['maximize' for i in range(7)]

    study = optuna.create_study(directions=directions)

else:

    with open('dp_SUPPORT.pkl', 'rb') as f:
        study = pickle.load(f)

study.optimize(
    lambda trial : objective(
    trial, gower=GenericBrowser, differential_privacy=differential_privacy, target_delta=target_delta, target_eps=target_eps, n_epochs=n_epochs
    ), n_trials=3, gc_after_trial=True
    ) # GC to avoid OOM
#%%

study.best_trials

#%% -------- Save The  Study -------- #

# For a multi objective study we need to find the best trials and basically
# average between the 3 metrics to get the best trial

with open("dp_SUPPORT.pkl", 'wb') as f:
        pickle.dump(study, f)

trial_averages = []

for trials in study.best_trials:

    metrics = trials.values
    trial_averages.append(np.mean(metrics))