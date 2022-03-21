import argparse
import warnings

# Standard imports
import numpy as np
import pandas as pd

# For the SUPPORT dataset
from pycox.datasets import support

# SDV aspects
# from sdgym.synthesizers import Independent

# from sdv.demo import load_tabular_demo
from sdv.evaluation import evaluate
from sdv.tabular import CopulaGAN, CTGAN, GaussianCopula, TVAE

# Other
from utils import set_seed, support_pre_proc, metric_calculation, reverse_transformers


warnings.filterwarnings("ignore")
set_seed(0)

MODEL_CLASSES = {
    "CopulaGAN": CopulaGAN,
    "CTGAN": CTGAN,
    "GaussianCopula": GaussianCopula,
    "TVAE": TVAE,
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_runs",
    default=10,
    type=int,
    help="set number of runs/seeds",
)
parser.add_argument(
    "--model_type",
    default="GaussianCopula",
    choices=MODEL_CLASSES.keys(),
    type=str,
    help="set model for baseline experiment",
)

parser.add_argument(
    "--pre_proc_method",
    default="GMM",
    type=str,
    help="Pre-processing method for the dataset. Either GMM or standard. (Gaussian mixture modelling method or standard scaler)"
)

args = parser.parse_args()

n_seeds = args.n_runs
my_seeds = np.random.randint(1e6, size=n_seeds)


data_supp = support.read_df()

# Setup column data types for SDV models to leverage

field_types = {
    "event" : "categorical",
    "x1" : "categorical",
    "x2" : "categorical",
    "x3" : "categorical",
    "x4" : "categorical",
    "x5" : "categorical",
    "x6" : "categorical"
}

#%% -------- Data Pre-Processing -------- #

pre_proc_method = args.pre_proc_method

data, data_supp, reordered_dataframe_columns, continuous_transformers, categorical_transformers, num_categories, num_continuous = support_pre_proc(data_supp=data_supp, pre_proc_method=pre_proc_method)

# Define lists to contain the metrics achieved on the
# train/generate/evaluate runs
svcs = []
gmlls = []
cs = []
ks = []
kses = []
contkls = []
disckls = []


# Perform the train/generate/evaluate runs
for i in range(n_seeds):
    set_seed(my_seeds[i])

    chosen_model = MODEL_CLASSES[args.model_type]

    model = chosen_model() #field_transformers=transformer_dtypes)

    print(
        f"Train + Generate + Evaluate {args.model_type}"
        f" - Run {i+1}/{n_seeds}"
    )

    model.fit(data)

    new_data = model.sample(data.shape[0])

    # new_data = Independent._fit_sample(data, None)

    data_ = data.copy()

    # Reverse the transformations

    synthetic_supp = reverse_transformers(synthetic_set=new_data, data_supp_columns=data_supp.columns, 
                                      cont_transformers=continuous_transformers, cat_transformers=categorical_transformers,
                                      pre_proc_method=pre_proc_method)

    evals = evaluate(new_data, data_, aggregate=False)

    svcs.append(np.array(evals["raw_score"])[2])
    gmlls.append(np.array(evals["raw_score"])[3])
    cs.append(np.array(evals["raw_score"])[4])
    ks.append(np.array(evals["raw_score"])[5])
    kses.append(np.array(evals["raw_score"])[6])
    contkls.append(np.array(evals["raw_score"])[7])
    disckls.append(np.array(evals["raw_score"])[8])

svcs = np.array(svcs)
gmlls = np.array(gmlls)
cs = np.array(cs)
ks = np.array(ks)
kses = np.array(kses)
contkls = np.array(contkls)
disckls = np.array(disckls)

print(f"SVC: {np.mean(svcs)} +/- {np.std(svcs)}")
print(f"GMLL: {np.mean(gmlls)} +/- {np.std(gmlls)}")
print(f"CS: {np.mean(cs)} +/- {np.std(cs)}")
print(f"KS: {np.mean(ks)} +/- {np.std(ks)}")
print(f"KSE: {np.mean(kses)} +/- {np.std(kses)}")
print(f"ContKL: {np.mean(contkls)} +/- {np.std(contkls)}")
print(f"DiscKL: {np.mean(disckls)} +/- {np.std(disckls)}")
