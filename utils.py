from datetime import datetime
from lzma import CHECK_CRC32
import numpy as np
import torch
from rdt.transformers import numerical, categorical


def set_seed(seed):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

# Pre-Processing for SUPPORT

def support_pre_proc(data_supp):

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

    continuous_columns = ['duration'] + [f"x{i}" for i in range(7,15)]
    categorical_columns = ['event'] + [f"x{i}" for i in range(1,7)] 
    num_categories = (
        np.array([np.amax(data_supp[col]) for col in categorical_columns]) + 1
    ).astype(int)
    num_continuous = len(continuous_columns)

    transformed_dataset = data_supp

    # Define columns based on datatype and then loop over creating and fitting 
    # transformers

    for index, column in enumerate(continuous_columns):

        # Fit GMM
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

    return x_train, reordered_dataframe.columns, continuous_transformers, categorical_transformers, num_categories, num_continuous