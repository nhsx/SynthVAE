import argparse
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

# VAE functions
from VAE import Decoder, Encoder, VAE

# Plotting
import matplotlib
font = {'size': 14}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# For the SUPPORT dataset
from pycox.datasets import support

# VAE functions
from VAE import Decoder, Encoder, VAE

parser = argparse.ArgumentParser()

parser.add_argument(
    "--savefile",
    required=True,
    type=str,
    help="load trained model's state_dict from file",
)

args = parser.parse_args()

# Import and preprocess the SUPPORT data for ground truth correlations
data_supp = support.read_df()
data_supp['x14'] = data_supp['x0']
# data_supp = data_supp.astype('float32')
data_supp = data_supp[['duration'] + [f'x{i}' for i in range(1,15)] + ['event']]
data_supp[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'event']] = data_supp[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'event']].astype(int)
transformer_dtypes = {
        'i': 'one_hot_encoding',
        'f': 'numerical',
        'O': 'one_hot_encoding',
        'b': 'one_hot_encoding',
        'M': 'datetime',
    }
ht = HyperTransformer(dtype_transformers=transformer_dtypes)
ht.fit(data_supp)
transformed = ht.transform(data_supp)
cat_cols = [f'x{i}' for i in range(1,7)] + ['event']
cont_cols = [f'x{i}' for i in range(7,15)] + ['duration']
num_categories = (np.array([np.amax(data_supp[col]) for col in cat_cols]) + 1).astype(int)
num_continuous = len(cont_cols)
cols_standardize = transformed.columns[:num_continuous]
cols_leave = transformed.columns[num_continuous:]
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [([col], None) for col in cols_leave]
x_mapper = DataFrameMapper(leave + standardize)
x_train_df = x_mapper.fit_transform(transformed)
x_train_df = x_mapper.transform(transformed)
x_train = x_train_df.astype('float32')
###############################################################################

# Load saved model
latent_dim = 2
encoder = Encoder(x_train.shape[1], latent_dim)
decoder = Decoder(latent_dim, num_continuous, num_categories=num_categories)
vae = VAE(encoder, decoder)
vae.load(args.savefile)

#  Collect samples and original data and transform them out standardised form
samples_ = vae.generate(data_supp.shape[0]).detach().numpy()
samples = np.ones_like(samples_)
samples[:,:num_continuous] = samples_[:,-num_continuous:]
samples[:,num_continuous:] = samples_[:,:-num_continuous]
samples = pd.DataFrame(samples)
samples.columns = transformed.columns
samples = ht.reverse_transform(samples)
t = x_mapper.transform(transformed)
t_ = np.ones_like(t)
t_[:,:num_continuous] = t[:,-num_continuous:]
t_[:,num_continuous:] = t[:,:-num_continuous]
t_ = pd.DataFrame(t_)
t_.columns = transformed.columns
t_ = ht.reverse_transform(t_)
samples[cat_cols] = samples[cat_cols].astype(object)
t_[cat_cols] = t_[cat_cols].astype(object)
for l in x_mapper.features:
  if l[0][0] in cont_cols:
    f = l[0][0]
    samples[f] = l[1].inverse_transform(samples[f])
    t_[f] = l[1].inverse_transform(t_[f])


### Create plots
# Plot 1: Correlation matrix of original data
plt.figure()
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
im = ax.matshow(t_.corr())
#####
# Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#####
plt.colorbar(im,cax=cax)
plt.savefig('actual_corr.png', bbox_inches='tight')
# Plot 2: Correlation matrix of synthetic data
plt.figure()
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
im = ax.matshow(samples.corr())
#####
# Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#####
plt.colorbar(im,cax=cax)
plt.savefig('sample_corr.png', bbox_inches='tight')
# Plot 3: Difference between real and synth correlation matrices + Gower and RMSE values
plt.figure()
g = np.mean(gower.gower_matrix(t_, samples))
p = np.sqrt(np.mean((t_.corr().to_numpy() - samples.corr().to_numpy())**2))
plt.title(f'Gower Distance = {g:.4f}\n Correlation RMSE = {p:.4f}')
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
im = ax.matshow(samples.corr() - t_.corr())
#####
# Credit: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
#####
plt.colorbar(im,cax=cax)
plt.savefig('diff_corr.png', bbox_inches='tight')
