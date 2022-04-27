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
from sdv.tabular import CopulaGAN, CTGAN, GaussianCopula, TVAE

# Other
from utils import set_seed, support_pre_proc, reverse_transformers
from metrics import distribution_metrics


warnings.filterwarnings("ignore")  # We suppress warnings to avoid SDMETRICS throwing unique synthetic data warnings (i.e.
# data in synthetic set is not in the real data set) as well as SKLEARN throwing convergence warnings (pre-processing uses
# GMM from sklearn and this throws non convergence warnings)

set_seed(0)

MODEL_CLASSES = {
    "CopulaGAN": CopulaGAN,
    "CTGAN": CTGAN,
    "GaussianCopula": GaussianCopula,
    "TVAE": TVAE,
}

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_runs", default=10, type=int, help="set number of runs/seeds",
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
    help="Pre-processing method for the dataset. Either GMM or standard. (Gaussian mixture modelling method or standard scaler)",
)

parser.add_argument(
    "--save_metrics",
    default=False,
    type=bool,
    help="Set if we want to save the metrics - saved under Metric Breakdown.csv unless changed",
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


data_supp = support.read_df()

# Setup columns

original_continuous_columns = ["duration"] + [f"x{i}" for i in range(7, 15)]
original_categorical_columns = ["event"] + [f"x{i}" for i in range(1, 7)]


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

data = pd.DataFrame(x_train, columns=reordered_dataframe_columns)

# Define distributional metrics required - for sdv_baselines this is set by default
distributional_metrics = [
    "SVCDetection",
    "GMLogLikelihood",
    "CSTest",
    "KSTest",
    "KSTestExtended",
    "ContinuousKLDivergence",
    "DiscreteKLDivergence",
]

# Define lists to contain the metrics achieved on the
# train/generate/evaluate runs
svc = []
gmm = []
cs = []
ks = []
kses = []
contkls = []
disckls = []

if args.gower:
    gowers = []


# Perform the train/generate/evaluate runs
for i in range(n_seeds):
    set_seed(my_seeds[i])

    chosen_model = MODEL_CLASSES[args.model_type]

    model = chosen_model()  # field_transformers=transformer_dtypes)

    print(f"Train + Generate + Evaluate {args.model_type}" f" - Run {i+1}/{n_seeds}")

    model.fit(data)

    new_data = model.sample(data.shape[0])

    # new_data = Independent._fit_sample(data, None)

    data_ = data.copy()

    # Reverse the transformations

    synthetic_supp = reverse_transformers(
        synthetic_set=new_data,
        data_supp_columns=data_supp.columns,
        cont_transformers=continuous_transformers,
        cat_transformers=categorical_transformers,
        pre_proc_method=pre_proc_method,
    )

    metrics = distribution_metrics(
        gower_bool=args.gower,
        distributional_metrics=distributional_metrics,
        data_supp=data_supp,
        synthetic_supp=synthetic_supp,
        categorical_columns=original_categorical_columns,
        continuous_columns=original_continuous_columns,
        saving_filepath=None,
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
    if args.gower:
        gowers.append(np.array(list_metrics[7]))

svc = np.array(svc)
gmm = np.array(gmm)
cs = np.array(cs)
ks = np.array(ks)
kses = np.array(kses)
contkls = np.array(contkls)
disckls = np.array(disckls)

if args.gower:

    gowers = np.array(gowers)
    print(f"Gowers: {np.mean(gowers)} +/- {np.std(gowers)}")

print(f"SVC: {np.mean(svc)} +/- {np.std(svc)}")
print(f"GMM: {np.mean(gmm)} +/- {np.std(gmm)}")
print(f"CS: {np.mean(cs)} +/- {np.std(cs)}")
print(f"KS: {np.mean(ks)} +/- {np.std(ks)}")
print(f"KSE: {np.mean(kses)} +/- {np.std(kses)}")
print(f"ContKL: {np.mean(contkls)} +/- {np.std(contkls)}")
print(f"DiscKL: {np.mean(disckls)} +/- {np.std(disckls)}")

if args.save_metrics:

    if args.gower:
        metrics = pd.DataFrame(
            {
                "SVCDetection": svc[:, 0],
                "GMLogLikelihood": gmm[:, 0],
                "CSTest": cs[:, 0],
                "KSTest": ks[:, 0],
                "KSTestExtended": kses[:, 0],
                "ContinuousKLDivergence": contkls[:, 0],
                "DiscreteKLDivergence": disckls[:, 0],
                "Gower": gowers[:, 0],
            }
        )
    else:
        metrics = pd.DataFrame(
            {
                "SVCDetection": svc[:, 0],
                "GMLogLikelihood": gmm[:, 0],
                "CSTest": cs[:, 0],
                "KSTest": ks[:, 0],
                "KSTestExtended": kses[:, 0],
                "ContinuousKLDivergence": contkls[:, 0],
                "DiscreteKLDivergence": disckls[:, 0],
            }
        )

    metrics.to_csv("Metric Breakdown.csv")  # Change filepath location here
