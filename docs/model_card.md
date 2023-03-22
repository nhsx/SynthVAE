# Model Card: Variational AutoEncoder with Differential Privacy

Following [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from
Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf), we're providing some information about about the Variational AutoEncoder (VAE) with Differential Privacy within this repository.

## Model Details

The implementation of the Variational AutoEncoder (VAE) with Differential Privacy within this repository was created as part of an NHSX Analytics Unit PhD internship project undertaken by Dominic Danks [(last commit to the repository: commit 88a4bdf)](https://github.com/nhsx/SynthVAE/commit/88a4bdf613f538af45834f22d38e52312cfe24c5). This model card describes the updated version of the model, released in March 2022.  Further information about the previous version created by Dom and its model implementation can be found in Section 5.4 of the associated [report](./reports/report.pdf).

## Model Use

### Intended Use

This model is intended for use in experimenting with the interplay of differential privacy and VAEs.

### Out-of-Scope Use Cases

This model is not suitable to provide privacy guarantees in a production environment.

## Training Data

Experiments in this repository are run against the [Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) dataset](https://biostat.app.vumc.org/wiki/Main/SupportDesc) accessed via the [pycox](https://github.com/havakv/pycox) python library. We also performed further analysis on a single table that we extracted from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/).

## Performance and Limitations

A from-scratch VAE implementation was compared against various models available within the [SDV](https://sdv.dev/) framework using a variety of quality and privacy metrics on the SUPPORT dataset. The VAE was found to be competitive with all of these models across the various metrics. Differential Privacy (DP) was introduced via [DP-SGD](https://dl.acm.org/doi/10.1145/2976749.2978318) and the performance of the VAE for different levels
of privacy was evaluated. It was found that as the level of Differential Privacy introduced by
DP-SGD was increased, it became easier to distinguish between synthetic and real data.

Proper evaluation of quality and privacy of synthetic data is challenging. In this work, we
utilised metrics from the SDV library due to their natural integration with the rest of the codebase.
A valuable extension of this work would be to apply a variety of external metrics, including
more advanced adversarial attacks to more thoroughly evaluate the privacy of the considered methods,
including as the level of DP is varied. It would also be of interest to apply DP-SGD and/or
[PATE](https://arxiv.org/pdf/1610.05755.pdf) to all of the considered methods and evaluate
whether the performance drop as a function of implemented privacy is similar or different
across the models.

Currently the SynthVAE model only works for data which is 'clean'. I.e data that has no missingness or NaNs within its input. It can handle continuous, categorical and datetime variables. Special types such as nominal data cannot be handled properly however the model may still run. Column names have to be specified in the code for the variable group they belong to. A simple example CSV file is provided under [example_input.csv](./example_input.csv).

Hyperparameter tuning of the model can result in errors if certain parameter values are selected. Most commonly, changing learning rate in our example results in errors during training. An extensive test to evaluate plausible ranges has not been performed as of yet. If you get errors during tuning then consider your hyperparameter values and adjust accordingly.
