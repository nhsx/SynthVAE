#%% -------- Import Libraries -------- #

# Standard imports
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../')

# For Gower distance
import gower

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

from rdt.transformers import categorical, numerical
from sklearn.preprocessing import QuantileTransformer

from utils import support_pre_proc

import optuna

# Load in the support data
data_supp = support.read_df()

# Save the original columns

original_continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
original_categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 

original_columns = original_categorical_columns + original_continuous_columns
#%% -------- Data Pre-Processing -------- #

x_train, data_supp, reordered_dataframe_columns, continuous_transformers, categorical_transformers, num_categories, num_continuous = support_pre_proc(data_supp=data_supp)

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

def objective(trial, differential_privacy=False. target_delta=1e-3, target_eps=10.0, n_epochs=50):

    latent_dim = 256 # Hyperparam
    hidden_dim = 256 # Hyperparam
    encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
    decoder = Decoder(
        latent_dim, num_continuous, num_categories=num_categories
    )

    vae = VAE(encoder, decoder, lr=1e-3) # lr hyperparam

    target_delta = target_delta
    target_eps = target_eps

    n_epochs = n_epochs

    if differential_privacy == True:
        log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.diff_priv_train(
            data_loader,
            n_epochs=n_epochs,
            C=10, # Hyperparam
            target_eps=target_eps,
            target_delta=target_delta, 
            sample_rate=sample_rate, # Hyperparam
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")

    else:

        log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.train(data_loader, n_epochs=n_epochs)

    # -------- Generate Synthetic Data -------- #

    # Generate a synthetic set using trained vae

    synthetic_trial = vae.generate(data_supp.shape[0]) # 8873 is size of support

    # -------- Inverse Transformation On Synthetic Trial -------- #

    # First add the old columns to the synthetic set to see what corresponds to what

    synthetic_dataframe = pd.DataFrame(synthetic_trial.detach().numpy(),  columns=reordered_dataframe_columns)

    # Now all of the transformations from the dictionary - first loop over the categorical columns

    synthetic_transformed_set = synthetic_dataframe

    for transformer_name in categorical_transformers:

        transformer = categorical_transformers[transformer_name]
        column_name = transformer_name[12:]

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    for transformer_name in continuous_transformers:

        transformer = continuous_transformers[transformer_name]
        column_name = transformer_name[11:]

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    # -------- SDV Metrics -------- #
    # Calculate the sdv metrics for SynthVAE

    # Define lists to contain the metrics achieved on the
    # train/generate/evaluate runs

    samples = synthetic_transformed_set

    # Need these in same column order

    samples = samples[data_supp.columns]

    # Now categorical columns need to be converted to objects as SDV infers data
    # types from the fields and integers/floats are treated as numerical not categorical

    samples[original_categorical_columns] = samples[original_categorical_columns].astype(object)
    data_supp[original_categorical_columns] = data_supp[original_categorical_columns].astype(object)

    evals = evaluate(samples, data_supp, metrics=['ContinuousKLDivergence', 'DiscreteKLDivergence'], aggregate=False)

    # New version has added a lot more evaluation metrics
    #bns = (np.array(evals["raw_score"])[0])
    #gmlls = (np.array(evals["raw_score"])[1])
    #cs = (np.array(evals["raw_score"])[2])
    #ks = (np.array(evals["raw_score"])[3])
    #kses = (np.array(evals["raw_score"])[4])
    contkls = (np.array(evals["raw_score"])[5])
    disckls = (np.array(evals["raw_score"])[6])
    gowers = (np.mean(gower.gower_matrix(data_supp, samples)))

    return [contkls, disckls, gowers]

study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'])
study.optimize(objective, n_trials=30, timeout=300)