#%%
import argparse
import warnings

# Standard imports
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
from sdv.evaluation import evaluate

from sdv.metrics.tabular import NumericalLR, NumericalMLP, NumericalSVR

#torch.cuda.is_available()
#torch.cuda.current_device()
#torch.cuda.get_device_name(0)

# Load in the support data
data_supp = support.read_df()

###############################################################################
# DATA PREPROCESSING #
# We one-hot the categorical cols and standardise the continuous cols
data_supp["x14"] = data_supp["x0"]
# data_supp = data_supp.astype('float32')
data_supp = data_supp[
    ["duration"] + [f"x{i}" for i in range(1, 15)] + ["event"]
]
data_supp[["x1", "x2", "x3", "x4", "x5", "x6", "event"]] = data_supp[
    ["x1", "x2", "x3", "x4", "x5", "x6", "event"]
].astype(int)
transformer_dtypes = {
    "i": "one_hot_encoding",
    "f": "numerical",
    "O": "one_hot_encoding",
    "b": "one_hot_encoding",
    "M": "datetime",
}
ht = HyperTransformer(dtype_transformers=transformer_dtypes)
ht.fit(data_supp)
transformed = ht.transform(data_supp)
cat_cols = [f"x{i}" for i in range(1, 7)] + ["event"]
cont_cols = [f"x{i}" for i in range(7, 15)] + ["duration"]
num_categories = (
    np.array([np.amax(data_supp[col]) for col in cat_cols]) + 1
).astype(int)
num_continuous = len(cont_cols)
cols_standardize = transformed.columns[:num_continuous]
cols_leave = transformed.columns[num_continuous:]
standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [([col], None) for col in cols_leave]
x_mapper = DataFrameMapper(leave + standardize)
x_train_df = x_mapper.fit_transform(transformed)
x_train_df = x_mapper.transform(transformed)
x_train = x_train_df.astype("float32")
###############################################################################
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

# Create VAE
latent_dim = 2
hidden_dim = 256
encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
decoder = Decoder(
    latent_dim, num_continuous, num_categories=num_categories
)
vae = VAE(encoder, decoder)

n_epochs = 50

log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.train(data_loader, n_epochs=n_epochs)

#%% -------- Plot Loss Features ELBO Breakdown -------- #

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = go.Figure()

x = np.arange(n_epochs)

fig.add_trace(go.Scatter(x=x, y=log_elbo, mode = "lines+markers", name = "ELBO"))

fig.add_trace(go.Scatter(x=x, y=log_reconstruction, mode = "lines+markers", name = "Reconstruction"))

fig.add_trace(go.Scatter(x=x, y=log_divergence, mode = "lines+markers", name = "Divergence"))

fig.update_layout(title="ELBO Breakdown",
    xaxis_title="Epochs",
    yaxis_title="Loss Value",
    legend_title="Loss",)

fig.show()

filepath_save = ''

# Save static image
fig.write_image("{}/ELBO Breakdown OLD.png".format(filepath_save))
# Save interactive image
fig.write_html("{}/ELBO Breakdown OLD.html".format(filepath_save))

#%% -------- Plot Loss Features Reconstruction Breakdown -------- #

# Initialize figure with subplots
fig = make_subplots(
    rows=1, cols=2, subplot_titles=("Categorical Likelihood", "Gaussian Likelihood")
)

# Add traces
fig.add_trace(go.Scatter(x=x, y=log_categorical, mode = "lines", name = "Categorical"), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=log_numerical, mode = "lines", name = "Numerical"), row=1, col=2)

# Update xaxis properties
fig.update_xaxes(title_text="Epochs", row=1, col=1)
fig.update_xaxes(title_text="Epochs", row=1, col=2)

# Update yaxis properties
fig.update_yaxes(title_text="Loss Value", row=1, col=1)

# Update title and height
fig.update_layout(title_text="Reconstruction Breakdown")

fig.show()

# Save static image
fig.write_image("{}/Reconstruction Breakdown OLD.png".format(filepath_save))
# Save interactive image
fig.write_html("{}/Reconstruction Breakdown OLD.html".format(filepath_save))
#%% -------- Generate Synthetic Data -------- #

#  Collect samples and transform them out of one-hot, standardised form
samples_ = vae.generate(data_supp.shape[0]).detach().numpy()
samples = np.ones_like(samples_)
samples[:, :num_continuous] = samples_[:, -num_continuous:]
samples[:, num_continuous:] = samples_[:, :-num_continuous]
samples = pd.DataFrame(samples)
samples.columns = transformed.columns
samples = ht.reverse_transform(samples)
t = x_mapper.transform(transformed)
t_ = np.ones_like(t)
t_[:, :num_continuous] = t[:, -num_continuous:]
t_[:, num_continuous:] = t[:, :-num_continuous]
t_ = pd.DataFrame(t_)
t_.columns = transformed.columns
t_ = ht.reverse_transform(t_)
samples[cat_cols] = samples[cat_cols].astype(object)
t_[cat_cols] = t_[cat_cols].astype(object)
for feature in x_mapper.features:
    if feature[0][0] in cont_cols:
        f = feature[0][0]
        samples[f] = feature[1].inverse_transform(samples[f])
        t_[f] = feature[1].inverse_transform(t_[f])

#%% -------- Plot Histograms For All The Variable Distributions -------- #

# Plot some examples using plotly

for column in cat_cols:

    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Synthetic {}".format(column), "Original {}".format(column))
    )

    # Add traces
    fig.add_trace(go.Histogram(x=samples[column], name = "Synthetic"), row=1, col=1)
    fig.add_trace(go.Histogram(x=t_[column], name = "Original"), row=1, col=2)

    # Update xaxis properties
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    # Update yaxis properties
    fig.update_yaxes(title_text="Counts", row=1, col=1)

    # Update title and height
    fig.update_layout(title_text="Variable {}".format(column))

    fig.show()

    # Save static image
    fig.write_image("{}/Variable {} OLD.png".format(filepath_save, column))
    # Save interactive image
    fig.write_html("{}/Variable {} OLD.html".format(filepath_save, column))

for column in cont_cols:
    
    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Synthetic {}".format(column), "Original {}".format(column))
    )

    # Add traces
    fig.add_trace(go.Histogram(x=samples[column], name = "Synthetic"), row=1, col=1)
    fig.add_trace(go.Histogram(x=t_[column], name = "Original"), row=1, col=2)

    # Update xaxis properties
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    # Update yaxis properties
    fig.update_yaxes(title_text="Counts", row=1, col=1)

    # Update title and height
    fig.update_layout(title_text="Variable {}".format(column))

    fig.show()

    # Save static image
    fig.write_image("{}/Variable {} OLD.png".format(filepath_save, column))
    # Save interactive image
    fig.write_html("{}/Variable {} OLD.html".format(filepath_save, column))

#%% -------- SDV Metrics -------- #

# Here we apply all of the SDV metrics to the SynthVAE updated model

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

evals = evaluate(samples, t_, aggregate=False)

bns.append(np.array(evals["raw_score"])[0])
lrs.append(np.array(evals["raw_score"])[1])
svcs.append(np.array(evals["raw_score"])[2])
gmlls.append(np.array(evals["raw_score"])[3])
cs.append(np.array(evals["raw_score"])[4])
ks.append(np.array(evals["raw_score"])[5])
kses.append(np.array(evals["raw_score"])[6])
contkls.append(np.array(evals["raw_score"])[7])
disckls.append(np.array(evals["raw_score"])[8])
gowers.append(np.mean(gower.gower_matrix(t_, samples)))

lr_priv = NumericalLR.compute(
    t_.fillna(0),
    samples.fillna(0),
    key_fields=(
        [f"x{i}" for i in range(1, data_supp.shape[1] - 2)]
        + ["event"]
        + ["duration"]
    ),
    sensitive_fields=["x14"],
)
lr_privs.append(lr_priv)

mlp_priv = NumericalMLP.compute(
    t_.fillna(0),
    samples.fillna(0),
    key_fields=(
        [f"x{i}" for i in range(1, data_supp.shape[1] - 2)]
        + ["event"]
        + ["duration"]
    ),
    sensitive_fields=["x14"],
)
mlp_privs.append(mlp_priv)

svr_priv = NumericalSVR.compute(
    t_.fillna(0),
    samples.fillna(0),
    key_fields=(
        [f"x{i}" for i in range(1, data_supp.shape[1] - 2)]
        + ["event"]
        + ["duration"]
    ),
    sensitive_fields=["x14"],
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

#%% --------Save These Metrics -------- #

# Save these metrics into a pandas dataframe

metrics = pd.DataFrame(data = [[bns,lrs,svcs,gmlls,cs,ks,kses,contkls,disckls,gowers]],
columns = ["BNLogLikelihood", "LogisticDetection", "SVCDetection", "GMLogLikelihood",
"CSTest", "KSTest", "KSTestExtended", "ContinuousKLDivergence", "DiscreteKLDivergence", "Gower"])

metrics.to_csv("{}/Metrics OLD.csv".format(filepath_save)
