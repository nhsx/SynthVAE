# Standard imports
import matplotlib.pyplot as plt
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

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from VAE import Decoder, Encoder, VAE

# SDV aspects
# from sdgym.synthesizers import Independent
from sdv.evaluation import evaluate
from sdv.tabular import TVAE, GaussianCopula, CTGAN, CopulaGAN
from sdv.metrics.tabular import NumericalLR, NumericalMLP, NumericalSVR

# Other
from utils import set_seed

set_seed(0)

n_seeds = 10
my_seeds = np.random.randint(1e6, size=n_seeds)

# Load in the support data
data_supp = support.read_df()

###############################################################################
### DATA PREPROCESSING ###
# We one-hot the categorical cols and standardise the continuous cols
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
# Prepare data for interaction with torch VAE
Y = torch.Tensor(x_train)
dataset = TensorDataset(Y)
batch_size = 32

generator=None
sample_rate = batch_size/len(dataset)
data_loader = torch.utils.data.DataLoader(
      dataset,
      batch_sampler=UniformWithReplacementSampler(
          num_samples=len(dataset),
          sample_rate=sample_rate,
          generator=generator
      ),
      pin_memory=True,
      generator=generator
  )

# shuffle = True
# data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle)

# Define lists to contain the metrics achieved on the
# train/generate/evaluate runs
bns = []
lrs = []
svcs = []
gmlls = []
cs = []
ks = []
kses = []
contkls = []
disckls = []
lr_privs = []
mlp_privs = []
svr_privs = []
gowers = []

target_delta = 1e-3
target_eps = 10.

for i in range(1):
  print(f"Train+Generate+Evaluate Run {i+1}/{n_seeds}")
  set_seed(my_seeds[i])

  # Create VAE
  latent_dim = 2
  encoder = Encoder(x_train.shape[1], latent_dim)
  decoder = Decoder(latent_dim, num_continuous, num_categories=num_categories)
  vae = VAE(encoder, decoder)

  vae.train(data_loader, n_epochs=60)

  # vae.diff_priv_train(data_loader, n_epochs=60, C=10, target_eps=target_eps, target_delta=target_delta, sample_rate=sample_rate)
  # print(vae.get_privacy_spent(target_delta))

  #  Collect samples and transform them out of one-hot, standardised form
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

  evals = evaluate(samples, t_, aggregate=False)
  bns.append(np.array(evals['raw_score'])[0])
  lrs.append(np.array(evals['raw_score'])[1])
  svcs.append(np.array(evals['raw_score'])[2])
  gmlls.append(np.array(evals['raw_score'])[3])
  cs.append(np.array(evals['raw_score'])[4])
  ks.append(np.array(evals['raw_score'])[5])
  kses.append(np.array(evals['raw_score'])[6])
  contkls.append(np.array(evals['raw_score'])[7])
  disckls.append(np.array(evals['raw_score'])[8])
  gowers.append(np.mean(gower.gower_matrix(t_, samples)))

  lr_priv = NumericalLR.compute(
      t_.fillna(0),
      samples.fillna(0),
      key_fields=([f'x{i}' for i in range(1,data_supp.shape[1]-2)] + ['event'] + ['duration']),
      sensitive_fields=['x14']
    )
  lr_privs.append(lr_priv)

  mlp_priv = NumericalMLP.compute(
    t_.fillna(0),
    samples.fillna(0),
    key_fields=([f'x{i}' for i in range(1,data_supp.shape[1]-2)] + ['event'] + ['duration']),
    sensitive_fields=['x14']
  )
  mlp_privs.append(mlp_priv)

  svr_priv = NumericalSVR.compute(
    t_.fillna(0),
    samples.fillna(0),
    key_fields=([f'x{i}' for i in range(1,data_supp.shape[1]-2)] + ['event'] + ['duration']),
    sensitive_fields=['x14']
  )
  svr_privs.append(svr_priv)

bns = np.array(bns)
lrs = np.array(lrs)
svcs = np.array(svcs)
gmlls = np.array(gmlls)
cs = np.array(cs)
ks = np.array(ks)
kses = np.array(kses)
contkls = np.array(contkls)
disckls = np.array(disckls)
gowers = np.array(gowers)
print(f"BN: {np.mean(bns)} +/- {np.std(bns)}")
print(f"LR: {np.mean(lrs)} +/- {np.std(lrs)}")
print(f"SVC: {np.mean(svcs)} +/- {np.std(svcs)}")
print(f"GMLL: {np.mean(gmlls)} +/- {np.std(gmlls)}")
print(f"CS: {np.mean(cs)} +/- {np.std(cs)}")
print(f"KS: {np.mean(ks)} +/- {np.std(ks)}")
print(f"KSE: {np.mean(kses)} +/- {np.std(kses)}")
print(f"ContKL: {np.mean(contkls)} +/- {np.std(contkls)}")
print(f"DiscKL: {np.mean(disckls)} +/- {np.std(disckls)}")
print(f"Gower: {np.mean(gowers)} +/- {np.std(gowers)}")

lr_privs = np.array(lr_privs)
print(f"LR privs: {np.mean(lr_privs)} +/- {np.std(lr_privs)}")
mlp_privs = np.array(mlp_privs)
print(f"MLP privs: {np.mean(mlp_privs)} +/- {np.std(mlp_privs)}")
svr_privs = np.array(svr_privs)
print(f"SVR privs: {np.mean(svr_privs)} +/- {np.std(svr_privs)}")
