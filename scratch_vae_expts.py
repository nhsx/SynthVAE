import argparse
import warnings

# Standard imports
import numpy as np
import pandas as pd
import torch

from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For the SUPPORT dataset
from pycox.datasets import support

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

# Other
from utils import (
    set_seed,
    support_pre_proc,
    plot_elbo,
    plot_likelihood_breakdown,
    plot_variable_distributions,
    reverse_transformers,
)
from metrics import distribution_metrics

warnings.filterwarnings("ignore")
set_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_runs", default=10, type=int, help="set number of runs/seeds",
)
parser.add_argument(
    "--diff_priv", default=False, type=bool, help="run VAE with differential privacy",
)
parser.add_argument(
    "--n_epochs", default=100, type=int, help="number of epochs to train for"
)
parser.add_argument(
    "--savemodel",
    default=None,
    type=bool,
    help="save trained model's state_dict to file",
)
parser.add_argument(
    "--savevisualisation",
    default=None,
    type=bool,
    help="save model visualisations for ELBO & variable generations - only applicable for final run",
)
parser.add_argument(
    "--savemetrics",
    default=None,
    type=bool,
    help="save metrics - averaged over all runs",
)
parser.add_argument(
    "--pre_proc_method",
    default="GMM",
    type=str,
    help="Pre-processing method for the dataset. Either GMM or standard. (Gaussian mixture modelling method or standard scaler)",
)
parser.add_argument(
    "--gower",
    default=False,
    type=bool,
    help="Do you want to calculate the average gower distance",
)

args = parser.parse_args()

n_seeds = args.n_runs
my_seeds = np.random.randint(1e6, size=n_seeds)
n_epochs = args.n_epochs

# Load in SUPPORT
data_supp = support.read_df()

# Save the original columns

original_continuous_columns = ["duration"] + [f"x{i}" for i in range(7, 15)]
original_categorical_columns = ["event"] + [f"x{i}" for i in range(1, 7)]

original_columns = original_categorical_columns + original_continuous_columns
#%% -------- Data Pre-Processing -------- #
pre_proc_method = args.pre_proc_method

(
    x_train,
    data_supp,
    reordered_dataframe_columns,
    continuous_transformers,
    categorical_transformers,
    num_categories,
    num_continuous,
) = support_pre_proc(data_supp=data_supp, pre_proc_method=pre_proc_method)

#%% Model Creation & Training

# Prepare data for interaction with torch VAE
Y = torch.Tensor(x_train)
dataset = TensorDataset(Y)

# User Parameters

# User defined hyperparams
# General training
batch_size = 32
latent_dim = 256
hidden_dim = 256
n_epochs = 5
logging_freq = 1  # Number of epochs we should log the results to the user
patience = 5  # How many epochs should we allow the model train to see if
# improvement is made
delta = 10  # The difference between elbo values that registers an improvement
filepath = None  # Where to save the best model


# Privacy params
differential_privacy = args.diff_priv  # Do we want to implement differential privacy
sample_rate = 0.1  # Sampling rate
C = 1e16  # Clipping threshold - any gradients above this are clipped
noise_scale = None  # Noise multiplier - influences how much noise to add
target_eps = 1  # Target epsilon for privacy accountant
target_delta = 1e-5  # Target delta for privacy accountant

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

# For metric saving - save each metric after each run for each seed
svc = []
gmm = []
cs = []
ks = []
kses = []
contkls = []
disckls = []

if args.gower == True:

    gowers = []

for i in range(n_seeds):
    diff_priv_in = ""
    if args.diff_priv:
        diff_priv_in = " with differential privacy"

    print(f"Train + Generate + Evaluate VAE{diff_priv_in} - Run {i+1}/{n_seeds}")
    set_seed(my_seeds[i])

    # Create VAE
    encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
    decoder = Decoder(latent_dim, num_continuous, num_categories=num_categories)
    vae = VAE(encoder, decoder)

    if differential_privacy == True:
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
            C=10,
            target_eps=target_eps,
            target_delta=target_delta,
            sample_rate=sample_rate,
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")
    else:
        (
            training_epochs,
            log_elbo,
            log_reconstruction,
            log_divergence,
            log_categorical,
            log_numerical,
        ) = vae.train(data_loader, n_epochs=n_epochs)

    #%% -------- Generate Synthetic Data -------- #

    # Generate a synthetic set using trained vae

    synthetic_trial = vae.generate(data_supp.shape[0])  # 8873 is size of support
    #%% -------- Inverse Transformation On Synthetic Trial -------- #

    synthetic_sample = vae.generate(data_supp.shape[0])

    if torch.cuda.is_available():
        synthetic_sample = pd.DataFrame(
            synthetic_sample.cpu().detach(), columns=reordered_dataframe_columns
        )
    else:
        synthetic_sample = pd.DataFrame(
            synthetic_sample.detach(), columns=reordered_dataframe_columns
        )

    # Reverse the transformations

    synthetic_supp = reverse_transformers(
        synthetic_set=synthetic_sample,
        data_supp_columns=data_supp.columns,
        cont_transformers=continuous_transformers,
        cat_transformers=categorical_transformers,
        pre_proc_method=pre_proc_method,
    )

    if args.savemodel == True:
        vae.save("SynthVAE model.pt")

    if args.savemetrics == True:

        metrics = distribution_metrics(
            gower_bool=args.gower,
            data_supp=data_supp,
            synthetic_supp=synthetic_supp,
            categorical_columns=original_categorical_columns,
            continuous_columns=original_continuous_columns,
            saving_filepath="",
            pre_proc_method=pre_proc_method,
        )

        list_metrics = [metrics[i] for i in metrics.columns]

        # New version has added a lot more evaluation metrics - only use fidelity metrics for now
        svc.append(np.array(list_metrics[0]))
        gmm.append(np.array(list_metrics[1]))
        cs.append(np.array(list_metrics[2]))
        ks.append(np.array(list_metrics[3]))
        kses.append(np.array(list_metrics[4]))
        contkls.append(np.array(list_metrics[5]))
        disckls.append(np.array(list_metrics[6]))
        if args.gower == True:
            gowers.append(np.array(list_metrics[7]))

if args.savemetrics == True:

    svc = np.array(svc)
    gmm = np.array(gmm)
    cs = np.array(cs)
    ks = np.array(ks)
    kses = np.array(kses)
    contkls = np.array(contkls)
    disckls = np.array(disckls)

    print(f"SVC: {np.mean(svc)} +/- {np.std(svc)}")
    print(f"GMM: {np.mean(gmm)} +/- {np.std(gmm)}")
    print(f"CS: {np.mean(cs)} +/- {np.std(cs)}")
    print(f"KS: {np.mean(ks)} +/- {np.std(ks)}")
    print(f"KSE: {np.mean(kses)} +/- {np.std(kses)}")
    print(f"ContKL: {np.mean(contkls)} +/- {np.std(contkls)}")
    print(f"DiscKL: {np.mean(disckls)} +/- {np.std(disckls)}")

    if args.gower == True:
        gowers = np.array(gowers)
        print(f"Gowers : {np.mean(gowers)} +/- {np.std(gowers)}")

    # Save these metrics into a pandas dataframe

    metrics = pd.DataFrame(
        data=[[svc, gmm, cs, ks, kses, contkls, disckls, gowers]],
        columns=[
            "SVCDetection",
            "GMLogLikelihood",
            "CSTest",
            "KSTest",
            "KSTestExtended",
            "ContinuousKLDivergence",
            "DiscreteKLDivergence",
            "Gower",
        ],
    )

    metrics.to_csv("Metric Breakdown.csv")
#%% -------- Visualisation Figures -------- ##
if args.savevisualisation == True:

    # -------- Plot ELBO Breakdowns -------- #

    elbo_fig = plot_elbo(
        n_epochs=training_epochs,
        log_elbo=log_elbo,
        log_reconstruction=log_reconstruction,
        log_divergence=log_divergence,
        saving_filepath="",
    )

    # -------- Plot Reconstruction Breakdowns -------- #

    likelihood_fig = plot_likelihood_breakdown(
        n_epochs=training_epochs,
        log_categorical=log_categorical,
        log_numerical=log_numerical,
        saving_filepath="",
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
