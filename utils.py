from datetime import datetime
from lzma import CHECK_CRC32
import numpy as np
import torch
from rdt.transformers import numerical, categorical, DatetimeTransformer
import pandas as pd


def set_seed(seed):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)

# -------- Pre-Processing for SUPPORT -------- #

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

    return x_train, data_supp, reordered_dataframe.columns, continuous_transformers, categorical_transformers, num_categories, num_continuous

# -------- Pre-Processing for MIMIC sets -------- #
# Internal sets provided by NHSX - outside users will have to stick with SUPPORT set

def mimic_pre_proc(data_supp, version=1):

    # Specify column configurations

    original_categorical_columns = ['ETHNICITY', 'DISCHARGE_LOCATION', 'GENDER', 'FIRST_CAREUNIT', 'VALUEUOM', 'LABEL']
    original_continuous_columns = ['Unnamed: 0', 'ROW_ID', 'SUBJECT_ID', 'VALUE', 'age']
    
    if(version==1):
        original_datetime_columns = ['ADMITTIME', 'DISCHTIME', 'DOB', 'CHARTTIME']
    elif(version==2):
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
        temp_datetime = DatetimeTransformer()
        temp_datetime.fit(transformed_dataset, columns = column)
        datetime_transformers['datetime_{}'.format(column)] = temp_datetime

        transformed_dataset = temp_datetime.transform(transformed_dataset)
        # These newly fitted datetime columns now need to be scaled
        # And treated as a continuous variable
        continuous_columns += [str(column) +'.value']

    # WE NEED TO RETAIN THIS SET AS METRICS DO NOT EVALUATE WITH DATETIMES BUT THEY WILL EVALUATE
    # IF DATETIMES ARE IN A SECONDS FORMAT

    original_metric_set = transformed_dataset.copy()

    for index, column in enumerate(continuous_columns):

        # Fit GMM
        temp_continuous = numerical.BayesGMMTransformer()
        temp_continuous.fit(transformed_dataset, columns = column)
        continuous_transformers['continuous_{}'.format(column)] = temp_continuous

        categorical_columns += [str(column) + '.component']

        transformed_dataset = temp_continuous.transform(transformed_dataset)

    # Each numerical one gets a .normalized column + a .component column giving the mixture info
    # This too needs to be one hot encoded

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

    return x_train, original_metric_set, reordered_dataframe.columns, continuous_transformers, categorical_transformers, datetime_transformers, num_categories, num_continuous

# -------- Reverse Transformations -------- #

def reverse_transformers(synthetic_set, data_supp_columns, cont_transformers, cat_transformers, date_transformers):

    # Now all of the transformations from the dictionary - first loop over the categorical columns

    synthetic_transformed_set = synthetic_set

    for transformer_name in cat_transformers:

        transformer = cat_transformers[transformer_name]
        column_name = transformer_name[12:]

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    for transformer_name in cont_transformers:

        transformer = cont_transformers[transformer_name]
        column_name = transformer_name[11:]

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    for transformer_name in date_transformers:

        transformer = date_transformers[transformer_name]
        column_name = transformer_name[9:]

        synthetic_transformed_set = transformer.reverse_transform(synthetic_transformed_set)

    synthetic_transformed_set = pd.DataFrame(synthetic_transformed_set, columns = data_supp_columns)

    return synthetic_transformed_set

# -------- Constraint based sampling for MIMIC work -------- #

def constraint_sampling_mimic(n_rows, vae, reordered_cols, data_supp_columns, cont_transformers, cat_transformers, date_transformers, reverse_transformers=reverse_transformers, version=1):

    # n_rows - the number of rows we require

    synthetic_trial = vae.generate(n_rows) # Generate our big set

    synthetic_dataframe = pd.DataFrame(synthetic_trial.cpu().detach().numpy(),  columns=reordered_cols)

    # Reverse the transforms

    synthetic_dataframe = reverse_transformers(synthetic_dataframe, data_supp_columns, cont_transformers, cat_transformers, date_transformers)

    def initial_check(synthetic_dataframe, version=1):

        n_rows = synthetic_dataframe.shape[0]

        # First check which columns do not match the constraints and remove them
        for i in range(n_rows):
            # If there are any to drop
            if(synthetic_dataframe['DISCHTIME'][i] < synthetic_dataframe['ADMITTIME'][i] or (synthetic_dataframe['CHARTTIME'][i] < synthetic_dataframe['ADMITTIME'][i])
            or (synthetic_dataframe['age'][i] < 0) or (synthetic_dataframe['DOB'][i] < synthetic_dataframe['ADMITTIME'][i])):
            
                # Drop the row inplace
                synthetic_dataframe.drop([i], axis=0, inplace=True)

        return None

    # Now we need to generate & perform this check over and over until all rows match

def constraint_filtering(n_rows, vae, reordered_cols, data_supp_columns, cont_transformers, cat_transformers, date_transformers, reverse_transformers=reverse_transformers, version=1):

    # Generate samples
    synthetic_trial = vae.generate(n_rows)
    # Create pandas dataframe in column order
    synthetic_dataframe = pd.DataFrame(synthetic_trial.cpu().detach().numpy(),  columns=reordered_cols)

    # Reverse all the transformations ready for filtering
    synthetic_dataframe = reverse_transformers(synthetic_dataframe, data_supp_columns, cont_transformers, cat_transformers, date_transformers)

    # Function to filter out the constraints from the set - returns valid dataframe
    def constraint_check(synthetic_df):
                                # age greater than 0                   patient was discharged after being admitted           patient admitted after their date of birth         patient first chart after admit time
        valid_df = synthetic_df[(synthetic_df['age'] > 0) | (synthetic_df['DISCHTIME'] >= synthetic_df['ADMITTIME']) | (synthetic_df['ADMITTIME'] > synthetic_df['DOB']) | (synthetic_df['CHARTTIME'] >= synthetic_df['ADMITTIME'])]

        return valid_df

    # Do first check
    synthetic_dataframe = constraint_check(synthetic_dataframe)

    # Loop over returning a valid dataframe each time until we get a set that is big enough
    while(synthetic_dataframe.shape[0] != n_rows):

        rows_needed = n_rows - synthetic_dataframe.shape[0]


        # If we have too many, remove the required amount
        if(rows_needed < 0):

            rows_needed = np.arange(abs(rows_needed))

            # Drop the bottom rows_needed amount

            synthetic_dataframe.drop(rows_needed, axis=0, inplace=True)

        # Need to generate enough to fill the dataframe
        else:
            
            new_set = vae.generate(rows_needed)

            new_set = pd.DataFrame(new_set.cpu().detach().numpy(),  columns=reordered_cols)

            new_set = reverse_transformers(new_set, data_supp_columns, cont_transformers, cat_transformers, date_transformers)

            new_filtered_set = constraint_check(new_set)

            # Add this onto the original and re-run
            synthetic_dataframe = pd.concat([synthetic_dataframe, new_filtered_set])

    return synthetic_dataframe