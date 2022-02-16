#%% -------- Import Libraries -------- #

# Standard imports
import numpy as np
import pandas as pd
import torch

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

# Load in the support data
data_supp = support.read_df()
#%% -------- Data Pre-Processing -------- #

# We one-hot the categorical cols and standardise the continuous cols
data_supp["x14"] = data_supp["x0"]
# data_supp = data_supp.astype('float32')
data_supp = data_supp[
    ["duration"] + [f"x{i}" for i in range(1, 15)] + ["event"]
]
data_supp[["x1", "x2", "x3", "x4", "x5", "x6", "event"]] = data_supp[
    ["x1", "x2", "x3", "x4", "x5", "x6", "event"]
].astype(int)

# As of coding this, new version of RDT adds in GMM transformer which is what we require, however hyper transformers do not work as individual
# transformers take a 'columns' argument that can only allow for fitting of one column - so you need to loop over and create one for each column
# in order to fit the dataset - https://github.com/sdv-dev/RDT/issues/376

continuous_transformers = {}
categorical_transformers = {}

original_continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
original_categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 

continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 
num_categories = (
    np.array([np.amax(data_supp[col]) for col in categorical_columns]) + 1
).astype(int)
num_continuous = len(continuous_columns)

transformed_dataset = data_supp

# Define columns based on datatype and then loop over creating and fitting 
# transformers

# Do continuous first via either GMM/Quantile transform
transform = 'GMM'  # Quant or GMM
for index, column in enumerate(continuous_columns):

    # Fit quantile transformer
    if(transform == 'Quant'):

        # Pick number of quantiles - defaults to n_samples if number larger
        temp_continuous = QuantileTransformer(n_quantiles=100000, output_distribution='normal') 
        temp_column = transformed_dataset[column].values.reshape(-1, 1)
        temp_continuous.fit(temp_column)
        continuous_transformers['continuous_{}'.format(column)] = temp_continuous
        transformed_dataset[column] = (temp_continuous.transform(temp_column)).flatten()

    # Fit GMM
    elif(transform == 'GMM'):

        temp_continuous = numerical.BayesGMMTransformer()
        temp_continuous.fit(transformed_dataset, columns = column)
        continuous_transformers['continuous_{}'.format(column)] = temp_continuous

        transformed_dataset = temp_continuous.transform(transformed_dataset)

        # Each numerical one gets a .normalized column + a .component column giving the mixture info
        # This too needs to be one hot encoded

        categorical_columns += [str(column) + '.component']
        normalised_column = str(column) + '.component'

        # Let's retrieve the new categorical and continuous column names

        continuous_columns = ['duration.normalized'] + [f"x{i}.normalized" for i in range(7,15)]

        # For each categorical column we want to know the number of categories

        num_categories = (
        np.array([np.amax(transformed_dataset[col]) for col in categorical_columns]) + 1
        ).astype(int)

        num_continuous = len(continuous_columns)

    for index, column in enumerate(categorical_columns):

        temp_categorical = categorical.OneHotEncodingTransformer()
        temp_categorical.fit(transformed_dataset, columns = column)
        categorical_transformers['categorical_{}'.format(index)] = temp_categorical

        transformed_dataset = temp_categorical.transform(transformed_dataset)

# We need the dataframe in the correct format i.e. categorical variables first and in the order of
# num_categories with continuous variables placed after

reordered_dataframe = pd.DataFrame()

reordered_dataframe = transformed_dataset.iloc[:, num_continuous:]

reordered_dataframe = pd.concat([reordered_dataframe, transformed_dataset.iloc[:, :num_continuous]], axis=1)

x_train_df = reordered_dataframe.to_numpy()
x_train = x_train_df.astype("float32")

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

# Create VAE
latent_dim = 256
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

# Save static image
#fig.write_image("Plots/OLD V NEW 14-02-2022/ELBO Breakdown QUANT.png")
# Save interactive image
#fig.write_html("Plots/OLD V NEW 14-02-2022/ELBO Breakdown QUANT.html")
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
#fig.write_image("Plots/OLD V NEW 14-02-2022/Reconstruction Breakdown QUANT.png")
# Save interactive image
#fig.write_html("Plots/OLD V NEW 14-02-2022/Reconstruction Breakdown QUANT.html")
#%% -------- Generate Synthetic Data -------- #

# Generate a synthetic set using trained vae

synthetic_trial = vae.generate(data_supp.shape[0]) # 8873 is size of support
#%% -------- Inverse Transformation On Synthetic Trial -------- #

# First add the old columns to the synthetic set to see what corresponds to what

synthetic_dataframe = pd.DataFrame(synthetic_trial.detach().numpy(),  columns=reordered_dataframe.columns)

# Now all of the transformations from the dictionary - first loop over the categorical columns

synthetic_transformed_set = synthetic_dataframe

for transformer_name in categorical_transformers:

    transformer = categorical_transformers[transformer_name]
    column_name = transformer_name[12:]

    synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

for transformer_name in continuous_transformers:

    transformer = continuous_transformers[transformer_name]
    column_name = transformer_name[11:]

    if(transform == 'Quant'):
    
        synthetic_transformed_set[column_name] = transformer.inverse_transform(synthetic_transformed_set[column_name].values.reshape(-1, 1)).flatten()

    elif(transform == 'GMM'):

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

#%% -------- Plot Histograms For All The Variable Distributions -------- #

# Plot some examples using plotly

for column in original_categorical_columns:

    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Synthetic {}".format(column), "Original {}".format(column))
    )

    # Add traces
    fig.add_trace(go.Histogram(x=synthetic_transformed_set[column], name = "Synthetic"), row=1, col=1)
    fig.add_trace(go.Histogram(x=data_supp[column], name = "Original"), row=1, col=2)

    # Update xaxis properties
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    # Update yaxis properties
    fig.update_yaxes(title_text="Counts", row=1, col=1)

    # Update title and height
    fig.update_layout(title_text="Variable {}".format(column))

    fig.show()

    # Save static image
    #fig.write_image("Plots/OLD V NEW 14-02-2022/Variable {} QUANT.png".format(column))
    # Save interactive image
    #fig.write_html("Plots/OLD V NEW 14-02-2022/Variable {} QUANT.html".format(column))

for column in original_continuous_columns:
    
    # Initialize figure with subplots
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Synthetic {}".format(column), "Original {}".format(column))
    )

    # Add traces
    fig.add_trace(go.Histogram(x=synthetic_transformed_set[column], name = "Synthetic"), row=1, col=1)
    fig.add_trace(go.Histogram(x=data_supp[column], name = "Original"), row=1, col=2)

    # Update xaxis properties
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    # Update yaxis properties
    fig.update_yaxes(title_text="Counts", row=1, col=1)

    # Update title and height
    fig.update_layout(title_text="Variable {}".format(column))

    fig.show()

    # Save static image
    #fig.write_image("Plots/OLD V NEW 14-02-2022/Variable {} QUANT.png".format(column))
    # Save interactive image
    #fig.write_html("Plots/OLD V NEW 14-02-2022/Variable {} QUANT.html".format(column))

#%% -------- SDV Metrics -------- #
# Calculate the sdv metrics for SynthVAE

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

samples = synthetic_transformed_set

# Need these in same column order

samples = samples[data_supp.columns]

# Now categorical columns need to be converted to objects as SDV infers data
# types from the fields and integers/floats are treated as numerical not categorical

original_continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
original_categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 

samples[original_categorical_columns] = samples[original_categorical_columns].astype(object)
data_supp[original_categorical_columns] = data_supp[original_categorical_columns].astype(object)

evals = evaluate(samples, data_supp, aggregate=False)

# New version has added a lot more evaluation metrics
bns.append(np.array(evals["raw_score"])[0])
lrs.append(np.array(evals["raw_score"])[1])
svcs.append(np.array(evals["raw_score"])[2])
gmlls.append(np.array(evals["raw_score"])[11])
cs.append(np.array(evals["raw_score"])[12])
ks.append(np.array(evals["raw_score"])[13])
kses.append(np.array(evals["raw_score"])[14])
contkls.append(np.array(evals["raw_score"])[27])
disckls.append(np.array(evals["raw_score"])[28])
gowers.append(np.mean(gower.gower_matrix(data_supp, samples)))

lr_priv = NumericalLR.compute(
    data_supp.fillna(0),
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
    data_supp.fillna(0),
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
    data_supp.fillna(0),
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

#metrics.to_csv("Plots/OLD V NEW 14-02-2022/Metrics QUANT.csv")