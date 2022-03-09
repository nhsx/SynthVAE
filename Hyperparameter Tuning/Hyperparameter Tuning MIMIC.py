#%% -------- Import Libraries -------- #

# Standard imports
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../')

# For Gower distance
import gower

import pickle

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

from rdt.transformers import categorical, numerical, datetime
from sklearn.preprocessing import QuantileTransformer

from utils import mimic_pre_proc, constraint_sampling_mimic, pandas_filtering

import optuna

filepath = ""

# Load in the MIMIC dataset
data_supp = pd.read_csv(filepath)

# Save the original columns

original_categorical_columns = ['ETHNICITY', 'DISCHARGE_LOCATION', 'GENDER', 'FIRST_CAREUNIT', 'VALUEUOM', 'LABEL']
original_continuous_columns = ['Unnamed: 0', 'ROW_ID', 'SUBJECT_ID', 'VALUE', 'age']
original_datetime_columns = ['ADMITTIME', 'DISCHTIME', 'DOB', 'CHARTTIME']

# Drop DOD column as it contains NANS - for now

data_supp = data_supp.drop('DOD', axis = 1)

original_columns = original_categorical_columns + original_continuous_columns + original_datetime_columns
#%% -------- Data Pre-Processing -------- #

x_train, original_metric_set, reordered_dataframe_columns, continuous_transformers, categorical_transformers, datetime_transformers, num_categories, num_continuous = mimic_pre_proc(data_supp=data_supp, version=2)

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

# Create VAE - either DP preserving or not

differential_privacy = False

# -------- Define our Optuna trial -------- #

def objective(trial, differential_privacy=False, target_delta=1e-3, target_eps=10.0, n_epochs=50):

    latent_dim = trial.suggest_int('Latent Dimension', 2, 128, step=2) # Hyperparam
    hidden_dim = trial.suggest_int('Hidden Dimension', 32, 1024, step=32) # Hyperparam
    encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim, device=dev)
    decoder = Decoder(
        latent_dim, num_continuous, num_categories=num_categories, device=dev
    )

    lr = trial.suggest_float('Learning Rate', 1e-5, 1e-1, step=1e-5)
    vae = VAE(encoder, decoder) # lr hyperparam

    target_delta = target_delta
    target_eps = target_eps

    n_epochs = n_epochs

    C = trial.suggest_int('C', 10, 1e4, step=50)

    if differential_privacy == True:
        log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.diff_priv_train(
            data_loader,
            n_epochs=n_epochs,
            C=C, # Hyperparam
            target_eps=target_eps,
            target_delta=target_delta, 
            sample_rate=sample_rate,
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")

    else:

        log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.train(data_loader, n_epochs=n_epochs)

    # -------- Generate Synthetic Data -------- #

    # Generate a synthetic set using trained vae

    n_rows = data_supp.shape[0]
    synthetic_transformed_set = pandas_filtering(n_rows=n_rows, vae=vae, reordered_cols=reordered_dataframe_columns, 
    data_supp_columns=data_supp.columns, cont_transformers=continuous_transformers, cat_transformers=categorical_transformers, date_transformers=datetime_transformers)

    # -------- SDV Metrics -------- #
    # Calculate the sdv metrics for SynthVAE

    # Define lists to contain the metrics achieved on the
    # train/generate/evaluate runs

    samples = synthetic_transformed_set
    metric_set = data_supp.copy()

    # We now need to transform the datetime columns using datetime transformers
    for col in original_datetime_columns:

        # Fit datetime transformer - converts to seconds
        temp_datetime = datetime.DatetimeTransformer()
        temp_datetime.fit(samples, columns = col)
        samples = temp_datetime.transform(samples)
        temp_datetime.fit(metric_set, columns = col)
        metric_set = temp_datetime.transform(metric_set)


    # Need these in same column order as the datetime transformed mimic set

    samples = samples[metric_set.columns]

    # Now categorical columns need to be converted to objects as SDV infers data
    # types from the fields and integers/floats are treated as numerical not categorical

    samples[original_categorical_columns] = samples[original_categorical_columns].astype(object)
    metric_set[original_categorical_columns] = metric_set[original_categorical_columns].astype(object)

    evals = evaluate(samples, metric_set, metrics=['ContinuousKLDivergence', 'DiscreteKLDivergence'], aggregate=False)

    # New version has added a lot more evaluation metrics
    #bns = (np.array(evals["raw_score"])[0])
    #gmlls = (np.array(evals["raw_score"])[1])
    #cs = (np.array(evals["raw_score"])[2])
    #ks = (np.array(evals["raw_score"])[3])
    #kses = (np.array(evals["raw_score"])[4])
    contkls = (np.array(evals["raw_score"])[0])
    disckls = (np.array(evals["raw_score"])[1])
    #gowers = (np.mean(gower.gower_matrix(metric_set, samples)))

    return [contkls, disckls]

#%% -------- Run Hyperparam Optimisation -------- #

# If there is no study object in your folder then run and save the study so
# It can be resumed if needed

first_run=True  # First run indicates if we are creating a new hyperparam study

if(first_run==True):

    study = optuna.create_study(directions=['maximize', 'maximize'])

else:

    with open('no_dp_MIMIC.pkl', 'rb') as f:
        study = pickle.load(f)

study.optimize(objective, n_trials=10, gc_after_trial=True) # GC to avoid OOM
#%%

study.best_trials
#%% -------- Save The  Study -------- #

# For a multi objective study we need to find the best trials and basically
# average between the 3 metrics to get the best trial

with open("no_dp_MIMIC.pkl", 'wb') as f:
        pickle.dump(study, f)

trial_averages = []

for trials in study.best_trials:

    metrics = trials.values
    trial_averages.append(np.mean(metrics))

# Now find best trial

best_trial = np.amax(np.asarray(trial_averages))

#%% -------- Find params -------- #

study.best_trials[-1].params['Learning Rate']