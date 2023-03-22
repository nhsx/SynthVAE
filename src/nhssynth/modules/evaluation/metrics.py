import numpy as np
from sdv.evaluation import evaluate
import gower
import pandas as pd

from sdv.metrics.tabular import NumericalMLP, CategoricalSVM

# Distributional metrics - Check distribution differences between synthetic & original dataset as well as how
# Easy it is to discriminate them i.e. svc detection
def distribution_metrics(
    gower_bool,
    distributional_metrics,
    data_supp,
    synthetic_supp,
    categorical_columns,
    continuous_columns,
    saving_filepath="metrics/",
    pre_proc_method="GMM",
):

    # Define lists to contain the metrics achieved

    no_metrics = len(distributional_metrics)
    metrics = []

    # Need these in same column order

    synthetic_supp = synthetic_supp[data_supp.columns]

    # Now categorical columns need to be converted to objects as SDV infers data
    # types from the fields and integers/floats are treated as numerical not categorical

    synthetic_supp[categorical_columns] = synthetic_supp[categorical_columns].astype(
        object
    )
    data_supp[categorical_columns] = data_supp[categorical_columns].astype(object)

    evals = evaluate(
        synthetic_supp, data_supp, metrics=distributional_metrics, aggregate=False
    )

    # evals is a pandas dataframe of metrics - if we want to add a gower metric then we can
    # save this separately

    metrics = np.array(evals["raw_score"])

    if gower_bool == True:

        # Find the gower distance
        metrics = np.append(
            metrics, np.mean(gower.gower_matrix(data_supp, synthetic_supp))
        )

        metrics = pd.DataFrame(
            data=[metrics], columns=(distributional_metrics + ["Gower"])
        )

    else:

        metrics = pd.DataFrame(data=[metrics], columns=(distributional_metrics))

    # Save these metrics into a pandas dataframe - if the user wants to

    if saving_filepath != None:

        metrics.to_csv(
            "{}Metrics_SynthVAE_{}.csv".format(saving_filepath, pre_proc_method)
        )

    return metrics


# Build in some privacy metrics from SDV - TO DO!!!


def privacy_metrics(
    private_variable,
    data_supp,
    synthetic_supp,
    categorical_columns,
    continuous_columns,
    saving_filepath=None,
    pre_proc_method="GMM",
):

    if private_variable in continuous_columns:

        continuous_columns = [
            column for column in continuous_columns if column != private_variable
        ]

        mlp_priv = NumericalMLP.compute(
            data_supp.fillna(0),
            synthetic_supp.fillna(0),
            key_fields=(continuous_columns),
            sensitive_fields=[private_variable],
        )

        return mlp_priv

    elif private_variable in categorical_columns:

        categorical_columns = [
            column for column in categorical_columns if column != private_variable
        ]

        svm_priv = CategoricalSVM.compute(
            data_supp.fillna(0),
            synthetic_supp.fillna(0),
            key_fields=(categorical_columns),
            sensitive_fields=[private_variable],
        )

        return svm_priv


# Build in some fairness metrics (will have to find a library/code these ourselves) - TO DO!!!


def fairness_metrics(
    user_metrics,
    data_supp,
    synthetic_supp,
    categorical_columns,
    continuous_columns,
    saving_filepath=None,
    pre_proc_method="GMM",
):

    return None
