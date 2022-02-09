import argparse
import warnings

# Standard imports
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
# from sdgym.synthesizers import Independent

# from sdv.demo import load_tabular_demo
from sdv.evaluation import evaluate
from sdv.tabular import CopulaGAN, CTGAN, GaussianCopula, TVAE
from sdv.metrics.tabular import NumericalLR, NumericalMLP, NumericalSVR

#%% -------- Data Pre-processing -------- #

data_supp = support.read_df()

# Explicitly type the categorical variables of the SUPPORT dataset
support_cols = ["x1", "x2", "x3", "x4", "x5", "x6", "event"]

data_supp[support_cols] = data_supp[support_cols].astype(object)
data = data_supp

# Define categorical and continuous column labels
cat_cols = [f"x{i}" for i in range(1, 7)] + ["event"]
cont_cols = ["x0"] + [f"x{i}" for i in range(7, 14)] + ["duration"]

# If preprocess is True, then a StandardScaler is applied
# to the continuous variables
preprocess = True
if preprocess:
    from rdt.transformers import categorical, numerical, boolean, datetime

    continuous_transformers = {}
    categorical_transformers = {}
    boolean_transformers = {}
    datetime_transformers = {}

    continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
    categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 
    num_categories = (
        np.array([np.amax(data_supp[col]) for col in categorical_columns]) + 1
    ).astype(int)

    transformed_dataset = data_supp

    # Define columns based on datatype and then loop over creating and fitting transformers

    # Do continuous first via GMM as it gives a mixture column that then needs to be encoded OHE
    for index, column in enumerate(continuous_columns):

        temp_continuous = numerical.BayesGMMTransformer()
        temp_continuous.fit(transformed_dataset, columns = column)
        continuous_transformers['continuous_{}'.format(index)] = temp_continuous

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

#%% -------- SDV Baseline Tests -------- #

chosen_model = 

model = chosen_model(field_transformers=transformer_dtypes)

print(
    f"Train + Generate + Evaluate {args.model_type}"
    f" - Run {i+1}/{n_seeds}"
)

model.fit(data)

new_data = model.sample(data.shape[0])

# new_data = Independent._fit_sample(data, None)

data_ = data.copy()

if preprocess:
    for feature in x_mapper.features:
        if feature[0][0] in cont_cols:
            f = feature[0][0]
            new_data[f] = feature[1].inverse_transform(new_data[f])
            data_[f] = feature[1].inverse_transform(data_[f])

evals = evaluate(new_data, data_, aggregate=False)

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

bns.append(np.array(evals["raw_score"])[0])
lrs.append(np.array(evals["raw_score"])[1])
svcs.append(np.array(evals["raw_score"])[2])
gmlls.append(np.array(evals["raw_score"])[3])
cs.append(np.array(evals["raw_score"])[4])
ks.append(np.array(evals["raw_score"])[5])
kses.append(np.array(evals["raw_score"])[6])
contkls.append(np.array(evals["raw_score"])[7])
disckls.append(np.array(evals["raw_score"])[8])
gowers.append(np.mean(gower.gower_matrix(data_, new_data)))

lr_priv = NumericalLR.compute(
    data_.fillna(0),
    new_data.fillna(0),
    key_fields=(
        [f"x{i}" for j in range(1, data_.shape[1] - 2)]
        + ["event"]
        + ["duration"]
    ),
    sensitive_fields=["x0"],
)
lr_privs.append(lr_priv)

mlp_priv = NumericalMLP.compute(
    data_.fillna(0),
    new_data.fillna(0),
    key_fields=(
        [f"x{i}" for j in range(1, data_.shape[1] - 2)]
        + ["event"]
        + ["duration"]
    ),
    sensitive_fields=["x0"],
)
mlp_privs.append(mlp_priv)

svr_priv = NumericalSVR.compute(
    data_.fillna(0),
    new_data.fillna(0),
    key_fields=(
        [f"x{i}" for j in range(1, data_.shape[1] - 2)]
        + ["event"]
        + ["duration"]
    ),
    sensitive_fields=["x0"],
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