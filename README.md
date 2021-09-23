# Synthetic Data Exploration: Variational Autoencoders
[![Python v3.8](https://img.shields.io/badge/python-v3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

## About the Project

This repository holds code for the PhD internship project (previously known as Synthetic Data Generation - VAE) contextualising and investigating the potential use of Variational AutoEncoders (VAEs) for synthetic health data generation undertaken by Dominic Danks.

[Project Description - Synthetic Data Exploration: Variational Autoencoders](https://nhsx.github.io/nhsx-internship-projects/synthetic-data-exploration-vae/)

_**Note:** No data, public or private are shared in this repository._

### Project Stucture

- The main code is found in the root of the repository (see Usage below for more information)
- A copy of [Opacus](https://github.com/pytorch/opacus) (v0.14.0), a library for training PyTorch models with differential privacy, is contained within the repository - some additional features were added to make this version of the library compatible with the VAE setup and may be removed in the future
- The accompanying [report](https://github.com/nhsx/SynthVAE/blob/main/report.pdf) is also available in the repository

### Built With

- [PyTorch](https://github.com/pytorch)
- [SDV](https://github.com/sdv-dev/SDV)
- [Opacus](https://github.com/pytorch/opacus)

## Getting Started

### Installation

To get a local copy up and running follow these simple steps.

To clone the repo:

`git clone https://github.com/nhsx/SynthVAE.git`

To create a suitable environment:
- ```python -m venv synth_env```
- `source synth_env/bin/activate`
- `pip install -r requirements.txt`

## Usage

To reproduce the experiments contained in the report:
- `python sdv_baselines.py`
- `python scratch_vae_expts.py`

## Roadmap

See the [open issues](https://github.com/nhsx/SynthVAE/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](https://github.com/nhsx/SynthVAE/blob/main/CONTRIBUTING.md) for detailed guidance._

## License

Distributed under the MIT License. _See [LICENSE](https://github.com/nhsx/SynthVAE/blob/main/LICENSE) for more information._

## Contact

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [analytics-unit@nhsx.nhs.uk](mailto:analytics-unit@nhsx.nhs.uk)

## Acknowledgements


