# Changelog

All notable changes to this project will be documented in this file.

Instructions on how to update this Changelog are available in the `Updating the Changelog` section of the [`CONTRIBUTING.md`](./CONTRIBUTING.md).  This project follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased v2.0.0]

### Breaking Changes

- Added Gaussian Mixture Modelling as a pre-processing feature for non-gaussian continuous variables

### New Features

- Introduced hyperparameter tuning using Optuna
- Expanded SynthVAE use cases to include MIMIC-III dataset
- Introduced plotting functionality & training logging for SynthVAE training
- Expanded `argparse` selections to allow more user flexibility
- Added MIMIC-III pre-processing notebook
- Introduced `random_state` changes as well as other seed changes to allow for reproducibility of results

## [Unreleased v1.0.0]

### New Features 

- Added project from Dom's [(djdnx)](https://github.com/djdnx) working repository
- Added missing project files
- Added `argparse` approach to running experiments

### Fixed

- Fixed black and flake8 adherence


[Unreleased]: https://github.com/nhsx/SynthVAE/tree/main
