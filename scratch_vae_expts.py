import argparse
import warnings

# Standard imports
import numpy as np
import pandas as pd
import torch

# For Gower distance
import gower

# For data preprocessing
from rdt.transformers import categorical, numerical

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

# Graph Visualisation
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Other
from utils import set_seed, support_pre_proc


warnings.filterwarnings("ignore")
set_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_runs",
    default=10,
    type=int,
    help="set number of runs/seeds",
)
parser.add_argument(
    "--diff_priv",
    default=False,
    type=bool,
    help="run VAE with differential privacy",
)
parser.add_argument(
    "--n_epochs",
    default=100,
    type=int,
    help="number of epochs to train for"
)
parser.add_argument(
    "--savefile",
    default=None,
    type=str,
    help="save trained model's state_dict to file",
)
parser.add_argument(
    "--savevisualisation",
    default=None,
    type=str,
    help="save model visualisations for ELBO & variable generations at the following filepath - only applicable for final run"
)
parser.add_argument(
    "--metrics",
    default=None,
    type=str,
    help="save metrics to the following filepath - averaged over all runs"
)

args = parser.parse_args()

n_seeds = args.n_runs
my_seeds = np.random.randint(1e6, size=n_seeds)
n_epochs = args.n_epochs

# Load in SUPPORT
data_supp = support.read_df()

# Save the original columns

original_continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
original_categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 

original_columns = original_categorical_columns + original_continuous_columns
#%% -------- Data Pre-Processing -------- #

x_train, data_supp, reordered_dataframe_columns, continuous_transformers, categorical_transformers, num_categories, num_continuous = support_pre_proc(data_supp=data_supp)

#%% Model Creation & Training

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

# shuffle = True
# data_loader = DataLoader(
#     dataset, batch_size=batch_size, pin_memory=True, shuffle=shuffle
# )

# For metric saving
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
target_eps = 10.0

for i in range(n_seeds):
    diff_priv_in = ""
    if args.diff_priv:
        diff_priv_in = " with differential privacy"

    print(
        f"Train + Generate + Evaluate VAE{diff_priv_in} - Run {i+1}/{n_seeds}"
    )
    set_seed(my_seeds[i])

    # Create VAE
    latent_dim = 2
    hidden_dim = 256
    encoder = Encoder(x_train.shape[1], latent_dim, hidden_dim=hidden_dim)
    decoder = Decoder(
        latent_dim, num_continuous, num_categories=num_categories
    )
    vae = VAE(encoder, decoder)

    if args.diff_priv:
        log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.diff_priv_train(
            data_loader,
            n_epochs=n_epochs,
            C=10,
            target_eps=target_eps,
            target_delta=target_delta,
            sample_rate=sample_rate,
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")
    else:
        log_elbo, log_reconstruction, log_divergence, log_categorical, log_numerical = vae.train(data_loader, n_epochs=n_epochs)

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

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    if args.savefile is not None:
        vae.save(args.savefile)

    if args.metrics is not None:
        samples = synthetic_transformed_set

        # Need these in same column order

        samples = samples[data_supp.columns]

        # Now categorical columns need to be converted to objects as SDV infers data
        # types from the fields and integers/floats are treated as numerical not categorical

        original_continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
        original_categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 

        samples[original_categorical_columns] = samples[original_categorical_columns].astype(object)
        data_supp[original_categorical_columns] = data_supp[original_categorical_columns].astype(object)

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

if(args.metrics is not None):    

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

    # Save these metrics into a pandas dataframe

    metrics = pd.DataFrame(data = [[bns,lrs,svcs,gmlls,cs,ks,kses,contkls,disckls,gowers]],
    columns = ["BNLogLikelihood", "LogisticDetection", "SVCDetection", "GMLogLikelihood",
    "CSTest", "KSTest", "KSTestExtended", "ContinuousKLDivergence", "DiscreteKLDivergence", "Gower"])

    filepath = args.metrics
    metrics.to_csv("{}/Metric Breakdown.csv".format(filepath))
#%% -------- Visualisation Figures -------- ##
if(args.savevisualisation is not None):

    filepath = args.savevisualisation

    # -------- Plot ELBO Breakdowns -------- #

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
    fig.write_image("{}/ELBO Breakdown.png".format(filepath))
    # Save interactive image
    fig.write_html("{}/ELBO Breakdown.html".format(filepath))

    # -------- Plot Reconstruction Breakdowns -------- #

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
    fig.write_image("{}/Reconstruction Breakdown.png".format(filepath))
    # Save interactive image
    fig.write_html("{}/Reconstruction Breakdown.html".format(filepath))

    
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
        fig.write_image("{}/Variable {}.png".format(filepath, column))
        # Save interactive image
        fig.write_html("{}/Variable {}.html".format(filepath, column))

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
        fig.write_image("{}/Variable {}.png".format(filepath, column))
        # Save interactive image
        fig.write_html("{}/Variable {}.html".format(filepath, column))