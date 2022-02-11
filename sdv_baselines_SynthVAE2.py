#%%
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

data_supp["x14"] = data_supp["x0"]
# data_supp = data_supp.astype('float32')
data_supp = data_supp[
    ["duration"] + [f"x{i}" for i in range(1, 15)] + ["event"]
]
data_supp[["x1", "x2", "x3", "x4", "x5", "x6", "event"]] = data_supp[
    ["x1", "x2", "x3", "x4", "x5", "x6", "event"]
].astype(int)

data = data_supp

# Define categorical and continuous column labels
cat_cols = [f"x{i}" for i in range(1, 7)] + ["event"]
cont_cols = ["x0"] + [f"x{i}" for i in range(7, 14)] + ["duration"]

# If preprocess is True, then a StandardScaler is applied
# to the continuous variables
preprocess = False
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