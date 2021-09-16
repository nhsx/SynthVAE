# Standard imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# For Gower Distance
import gower

# For data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

# For the SUPPORT dataset
from pycox.datasets import support

# SDV aspects
from sdgym.synthesizers import Independent

from sdv.demo import load_tabular_demo
from sdv.evaluation import evaluate
from sdv.tabular import TVAE, GaussianCopula, CTGAN, CopulaGAN
from sdv.metrics.tabular import NumericalLR, NumericalMLP, NumericalSVR

# Other
from utils import set_seed

set_seed(0)

n_seeds = 10
my_seeds = np.random.randint(1e6, size=n_seeds)

data_supp = support.read_df()

# Explicitly type the categorical variables of the SUPPORT dataset
data_supp[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'event']] = data_supp[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'event']].astype(object)
data = data_supp

# Define categorical and continuous column labels
cat_cols = [f'x{i}' for i in range(1,7)] + ['event']
cont_cols = ['x0'] + [f'x{i}' for i in range(7,14)] + ['duration']

# If preprocess is True, then a StandardScaler is applied to the continuous variables
preprocess = True
if preprocess:
  standardize = [([col], StandardScaler()) for col in cont_cols]
  leave = [([col], None) for col in cat_cols]

  x_mapper = DataFrameMapper(leave + standardize)

  x_train_df = x_mapper.fit_transform(data)
  x_train_df = x_mapper.transform(data)

  data_ = pd.DataFrame(x_train_df)
  data_.columns = cat_cols + cont_cols
  data_[cont_cols] = data_[cont_cols].astype('float32')
  data = data_

transformer_dtypes = {
        'i': 'one_hot_encoding',
        'f': 'numerical',
        'O': 'one_hot_encoding',
        'b': 'one_hot_encoding',
        'M': 'datetime',
    }

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

# Perform the train/generate/evaluate runs
for i in range(n_seeds):
  print(f"Train+Generate+Evaluate Run {i+1}/{n_seeds}")
  set_seed(my_seeds[i])

  ########### CHANGE THE LINE BELOW TO CHANGE MODELS ###########
  model = GaussianCopula(field_transformers=transformer_dtypes)
  ##############################################################

  model.fit(data)

  new_data = model.sample(data.shape[0])

  # new_data = Independent._fit_sample(data, None)

  data_ = data.copy()

  if preprocess:
    for l in x_mapper.features:
      if l[0][0] in cont_cols:
        f = l[0][0]
        new_data[f] = l[1].inverse_transform(new_data[f])
        data_[f] = l[1].inverse_transform(data_[f])

  evals = evaluate(new_data, data_, aggregate=False)

  bns.append(np.array(evals['raw_score'])[0])
  lrs.append(np.array(evals['raw_score'])[1])
  svcs.append(np.array(evals['raw_score'])[2])
  gmlls.append(np.array(evals['raw_score'])[3])
  cs.append(np.array(evals['raw_score'])[4])
  ks.append(np.array(evals['raw_score'])[5])
  kses.append(np.array(evals['raw_score'])[6])
  contkls.append(np.array(evals['raw_score'])[7])
  disckls.append(np.array(evals['raw_score'])[8])
  gowers.append(np.mean(gower.gower_matrix(data_, new_data)))

  lr_priv = NumericalLR.compute(
    data_.fillna(0),
    new_data.fillna(0),
    key_fields=([f'x{i}' for j in range(1,data_.shape[1]-2)] + ['event'] + ['duration']),
    sensitive_fields=['x0']
  )
  lr_privs.append(lr_priv)

  mlp_priv = NumericalMLP.compute(
    data_.fillna(0),
    new_data.fillna(0),
    key_fields=([f'x{i}' for j in range(1,data_.shape[1]-2)] + ['event'] + ['duration']),
    sensitive_fields=['x0']
  )
  mlp_privs.append(mlp_priv)

  svr_priv = NumericalSVR.compute(
    data_.fillna(0),
    new_data.fillna(0),
    key_fields=([f'x{i}' for j in range(1,data_.shape[1]-2)] + ['event'] + ['duration']),
    sensitive_fields=['x0']
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
