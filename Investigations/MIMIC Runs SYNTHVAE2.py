#%% -------- Import Libraries -------- #

# Standard imports
from tokenize import String
import numpy as np
import pandas as pd
import torch

# For Gower distance
import gower

from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE is in other folder
import sys
sys.path.append('../')

# VAE functions
from VAE import Decoder, Encoder, VAE

# SDV aspects
from sdv.evaluation import evaluate

from sdv.metrics.tabular import NumericalLR, NumericalMLP, NumericalSVR

from rdt.transformers import categorical, numerical, datetime

# Graph Visualisation
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Load in the mimic single table data

filepath = ''

data_supp = pd.read_csv(filepath)
#%% -------- Data Pre-Processing -------- #

# Specify column configurations

original_categorical_columns = ['ETHNICITY', 'DISCHARGE_LOCATION', 'GENDER', 'FIRST_CAREUNIT', 'VALUEUOM', 'LABEL']
original_continuous_columns = ['Unnamed: 0', 'ROW_ID', 'SUBJECT_ID', 'VALUE', 'age']
original_datetime_columns = ['ADMITTIME', 'DISCHTIME', 'DOB', 'CHARTTIME']

categorical_columns = original_categorical_columns.copy()
continuous_columns = original_continuous_columns.copy()
datetime_columns = original_datetime_columns.copy()

# As of coding this, new version of RDT adds in GMM transformer which is what we require, however hyper transformers do not work as individual
# transformers take a 'columns' argument that can only allow for fitting of one column - so you need to loop over and create one for each column
# in order to fit the dataset - https://github.com/sdv-dev/RDT/issues/376

continuous_transformers = {}
categorical_transformers = {}
datetime_transformers = {}

# num_categories is either the maximum number within the categorical
# column, or the number of unique string terms

num_categories = []

for col in original_categorical_columns:

    if(data_supp[col].dtype == float):

        # Convert to int
        data_supp[col] = data_supp[col].astype(int)

    if(data_supp[col].dtype == int):

        num_categories.append(np.amax(data_supp[col]) + 1)

    # Categories are strings
    else:
        
        # Convert column into one type
        values= np.unique(data_supp[col].astype(str), return_counts=False)
        num_categories.append(values.shape[0])

num_continuous = len(original_continuous_columns)

transformed_dataset = data_supp

# Define columns based on datatype and then loop over creating and fitting 
# transformers

# Do datetime columns first to convert to seconds

for index, column in enumerate(original_datetime_columns):

    # Fit datetime transformer - converts to seconds
    temp_datetime = datetime.DatetimeTransformer()
    temp_datetime.fit(transformed_dataset, columns = column)
    datetime_transformers['datetime_{}'.format(column)] = temp_datetime

    transformed_dataset = temp_datetime.transform(transformed_dataset)
    # These newly fitted datetime columns now need to be scaled
    # And treated as a continuous variable
    continuous_columns += [str(column) +'.value']

for index, column in enumerate(continuous_columns):

    # Fit GMM
    temp_continuous = numerical.BayesGMMTransformer()
    temp_continuous.fit(transformed_dataset, columns = column)
    continuous_transformers['continuous_{}'.format(column)] = temp_continuous

    transformed_dataset = temp_continuous.transform(transformed_dataset)

    # Each numerical one gets a .normalized column + a .component column giving the mixture info
    # This too needs to be one hot encoded

    categorical_columns += [str(column) + '.component']


num_categories = []

continuous_columns = [str(col) + '.normalized' for col in continuous_columns]

for col in categorical_columns:

    if(transformed_dataset[col].dtype == float):

        # Convert to int
        transformed_dataset[col] = transformed_dataset[col].astype(int)

    if(transformed_dataset[col].dtype == int):

        num_categories.append(np.amax(transformed_dataset[col]) + 1)

    # Categories are strings/objects
    else:
        
        # Convert column into one type
        values = np.unique(transformed_dataset[col].astype(str), return_counts=False)

        num_categories.append(values.shape[0])

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

filepath_save = ''

# Save static image
fig.write_image("{}/ELBO Breakdown SynthVAE2.png".format(filepath_save))
# Save interactive image
fig.write_html("{}/ELBO Breakdown SynthVAE2.html".format(filepath_save))
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
fig.write_image("{}/Reconstruction Breakdown SYNTHVAE2.png".format(filepath_save))
# Save interactive image
fig.write_html("{}/Reconstruction Breakdown SYNTHVAE2.html".format(filepath_save))

#%% -------- Constraint Sampling -------- #
def reverse_transformers(synthetic_set, data_supp_columns, cont_transformers, cat_transformers, date_transformers):

    # Now all of the transformations from the dictionary - first loop over the categorical columns

    synthetic_transformed_set = synthetic_set

    for transformer_name in categorical_transformers:

        transformer = categorical_transformers[transformer_name]
        column_name = transformer_name[12:]

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    for transformer_name in continuous_transformers:

        transformer = continuous_transformers[transformer_name]
        column_name = transformer_name[11:]

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    for transformer_name in datetime_transformers:

        transformer = datetime_transformers[transformer_name]
        column_name = transformer_name[9:]

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    synthetic_transformed_set = pd.DataFrame(synthetic_transformed_set, columns = data_supp_columns)

    return synthetic_transformed_set

# 3 Big checks for this MIMIC Set:
# 1. Age be greater than 0
# 2. DISCHTIME should be after the ADMITTIME
# 3. CHARTTIME should also be after ADMITTIME

# Can speed this up by only passing in the amount of rows we need to create then passing those through the checks - only need to check whole df once
def constraint_sampling(n_rows, vae, reordered_cols, data_supp_columns, cont_transformers, cat_transformers, date_transformers, reverse_transformers):

    # n_rows - the number of rows we require

    synthetic_trial = vae.generate(n_rows) # Generate our big set

    synthetic_dataframe = pd.DataFrame(synthetic_trial.detach().numpy(),  columns=reordered_cols)

    # Reverse the transforms

    synthetic_dataframe = reverse_transformers(synthetic_dataframe, data_supp_columns, cont_transformers, cat_transformers, date_transformers)

    def initial_check(synthetic_dataframe):

        n_rows = synthetic_dataframe.shape[0]

        # First check which columns do not match the constraints and remove them
        for i in range(n_rows):
        
            # If there are any to drop
            if(synthetic_dataframe['DISCHTIME'][i] < synthetic_dataframe['ADMITTIME'][i] or (synthetic_dataframe['CHARTTIME'][i] < synthetic_dataframe['ADMITTIME'][i])
            or (synthetic_dataframe['age'][i] < 0)):
            
                # Drop the row inplace
                synthetic_dataframe.drop([i], axis=0, inplace=True)

        return None

    # Now we need to generate & perform this check over and over until all rows match

    def generation_checks(new_rows, vae, reordered_cols, initial_check, data_supp_columns, cont_transformers, cat_transformers, date_transformers):

        # Generate the amount we need
        new_samples = vae.generate(new_rows)

        new_dataframe = pd.DataFrame(new_samples.detach().numpy(), columns = reordered_cols) 

        # Reverse transforms
        synthetic_dataframe = reverse_transformers(new_dataframe, data_supp_columns, cont_transformers, cat_transformers, date_transformers)

        # Perform the first check

        initial_check(synthetic_dataframe)

        return synthetic_dataframe

    # First pass the generated set through the initial check to see if we need to do constraint sampling

    initial_check(synthetic_dataframe)

    # While synthetic_dataframe.shape[0] is not the amount we need (or we are racking up excessive attempts), we perform the loop
    n_tries = 0
    while( (synthetic_dataframe.shape[0] != n_rows) or (n_tries == 100) ):

        # Generate the amount required
        rows_needed = n_rows - synthetic_dataframe.shape[0]

        # Possible that we could have added extra rows to whats required so just remove these
        if(rows_needed < 0):
            
            rows_needed = np.arange(abs(rows_needed))

            # Drop the bottom rows_needed amount

            synthetic_dataframe.drop(rows_needed, axis=0, inplace=True)

        # We do not have enough rows so need to generate
        else:

            checked_rows = generation_checks(rows_needed, vae, reordered_cols, initial_check, data_supp_columns, cont_transformers, cat_transformers, date_transformers)

            # Add the rows that do fit constraints to the synthetic_dataframe
            synthetic_dataframe = pd.concat([synthetic_dataframe, checked_rows])
            
            n_tries += 1

    return synthetic_dataframe

#%% -------- Run Constraint Sampling -------- #

synthetic_transformed_set = constraint_sampling(
    n_rows=data_supp.shape[0], vae=vae, reordered_cols=reordered_dataframe.columns, data_supp_columns=data_supp.columns,
    cont_transformers=continuous_transformers, cat_transformers=categorical_transformers, date_transformers=datetime_transformers,
    reverse_transformers=reverse_transformers
)
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
    fig.write_image("{}/Variable {} CONSTRAINT.png".format(filepath_save, column))
    # Save interactive image
    fig.write_html("{}/Variable {} CONSTRAINT.html".format(filepath_save, column))

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
    fig.write_image("{}/Variable {} CONSTRAINT.png".format(filepath_save, column))
    # Save interactive image
    fig.write_html("{}/Variable {} CONSTRAINT.html".format(filepath_save, column))

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

samples[original_categorical_columns] = samples[original_categorical_columns].astype(object)
data_supp[original_categorical_columns] = data_supp[original_categorical_columns].astype(object)

# Remove datetime columns for now

metric_columns = original_categorical_columns + original_continuous_columns

samples = samples[metric_columns]
data_supp = data_supp[metric_columns]

evals = evaluate(samples, data_supp, metrics=['BNLogLikelihood','LogisticDetection','SVCDetection','GMLogLikelihood','CSTest','KSTest','KSTestExtended','ContinuousKLDivergence'
                                                , 'DiscreteKLDivergence'],aggregate=False)

# New version has added a lot more evaluation metrics - only use fidelity metrics for now
bns.append(np.array(evals["raw_score"])[0])
lrs.append(np.array(evals["raw_score"])[1])
svcs.append(np.array(evals["raw_score"])[2])
gmlls.append(np.array(evals["raw_score"])[3])
cs.append(np.array(evals["raw_score"])[4])
ks.append(np.array(evals["raw_score"])[5])
kses.append(np.array(evals["raw_score"])[6])
contkls.append(np.array(evals["raw_score"])[7])
disckls.append(np.array(evals["raw_score"])[8])
gowers.append(np.mean(gower.gower_matrix(data_supp, samples)))

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

metrics.to_csv("{}/Metrics CONSTRAINT.csv".format(filepath_save))
