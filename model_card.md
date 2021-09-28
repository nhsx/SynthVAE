# Model Card: Variational AutoEncoder with Differential Privacy

Following [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from
Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf), we're providing some information about about the Variational AutoEncoder (VAE) with Differential Privacy within this repository.

## Model Details

The implementation of the Variational AutoEncoder (VAE) with Differential Privacy within this repository was created as part of an NHSX Analytics Unit PhD internship project undertaken by Dominic Danks.  This model card describes the first version of the model, released in September 2021.  Further information about the model implementation can be found in Section 5.4 of the associated [report](./reports/report.pdf).

## Model Use

### Intended Use

This model is intended for use in experimenting with the interplay of differential privacy and VAEs.

### Out-of-Scope Use Cases

This model is not suitable to provide privacy guarentees in a production enviroment.

## Training Data

Experiments in this repository are run against the [Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) dataset](https://biostat.app.vumc.org/wiki/Main/SupportDesc) accessed via the [pycox](https://github.com/havakv/pycox) python library.

## Performance and Limitations

To be updated.