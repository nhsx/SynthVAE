import argparse
import numpy as np
import pandas as pd
import torch

# For Gower distance
import gower

# For the SUPPORT dataset
from pycox.datasets import support

# VAE functions
from VAE import Decoder, Encoder, VAE

from utils import support_pre_proc, reverse_transformers

# Plotting
import matplotlib

font = {"size": 14}
matplotlib.rc("font", **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# For the SUPPORT dataset
from pycox.datasets import support

# VAE functions
from VAE import Decoder, Encoder, VAE

parser = argparse.ArgumentParser()

parser.add_argument(
    "--save_file",
    required=True,
    type=str,
    help="load trained model's state_dict from file",
)

parser.add_argument(
    "--pre_proc_method",
    default="GMM",
    type=str,
    help="Choose the pre-processing method that you will apply to the dataset, either GMM or standard",
)

args = parser.parse_args()

# Import and preprocess the SUPPORT data for ground truth correlations
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


###############################################################################

# Load saved model - ensure parameters are equivalent to the saved model
latent_dim = 256
hidden_dim = 256
encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
decoder = Decoder(latent_dim, num_continuous, num_categories=num_categories)
vae = VAE(encoder, decoder)
vae.load(args.save_file)

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


### Create plots
# Plot 1: Correlation matrix of original data
plt.figure()
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
im = ax.matshow(data_supp.corr())
#####
# Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#####
plt.colorbar(im, cax=cax)
plt.savefig("actual_corr_{}.png".format(pre_proc_method), bbox_inches="tight")
# Plot 2: Correlation matrix of synthetic data
plt.figure()
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
im = ax.matshow(synthetic_supp.corr())
#####
# Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#####
plt.colorbar(im, cax=cax)
plt.savefig("sample_corr_{}.png".format(pre_proc_method), bbox_inches="tight")
# Plot 3: Difference between real and synth correlation matrices + Gower and RMSE values
plt.figure()
g = np.mean(gower.gower_matrix(data_supp, synthetic_supp))
p = np.sqrt(
    np.mean((data_supp.corr().to_numpy() - synthetic_supp.corr().to_numpy()) ** 2)
)
plt.title(f"Gower Distance = {g:.4f}\n Correlation RMSE = {p:.4f}")
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
im = ax.matshow(synthetic_supp.corr() - data_supp.corr())
#####
# Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#####
plt.colorbar(im, cax=cax)
plt.savefig("diff_corr_{}.png".format(pre_proc_method), bbox_inches="tight")
