# Synthetic Data Exploration: Variational Autoencoders
[![Python v3.8](https://img.shields.io/badge/python-v3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

## About the Project

This repository holds code for the PhD internship project (previously known as Synthetic Data Generation - VAE) contextualising and investigating the potential use of Variational AutoEncoders (VAEs) for synthetic health data generation undertaken by Dominic Danks.

[Project Description - Synthetic Data Exploration: Variational Autoencoders](https://nhsx.github.io/nhsx-internship-projects/synthetic-data-exploration-vae/)

_**Note:** No data, public or private are shared in this repository._

### Project Stucture

- The main code is found in the root of the repository (see Usage below for more information)
- The accompanying [report](./reports/report.pdf) is also available in the `reports` folder

**N.B.** A copy of [Opacus](https://github.com/pytorch/opacus) (v0.14.0), a library for training PyTorch models with differential privacy, is contained within the repository - some additional features were added to make this version of the library compatible with the VAE setup and may be removed in the future.

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

### SDV Baselines

To reproduce the experiments contained in the report involving the [SDV](https://github.com/sdv-dev/SDV) baseline models (e.g. CopulaGAN, CTGAN, GaussianCopula and TVAE), the parameters can be found using the `--help` flag:

```
python sdv_baselines.py --help

usage: sdv_baselines.py [-h] [--n_runs N_RUNS] [--model_type {CopulaGAN,CTGAN,GaussianCopula,TVAE}]

optional arguments:
  -h, --help            show this help message and exit
  --n_runs N_RUNS       set number of runs/seeds
  --model_type {CopulaGAN,CTGAN,GaussianCopula,TVAE}
                        set model for baseline experiment
```

### Scratch VAE + Differential Privacy

To reproduce the experiments contained in the report involving the VAE with/without differential privacy, the parameters can be found using the `--help` flag:

```
python scratch_vae_expts.py --help

usage: scratch_vae_expts.py [-h] [--n_runs N_RUNS] [--diff_priv DIFF_PRIV]

optional arguments:
  -h, --help            show this help message and exit
  --n_runs N_RUNS       set number of runs/seeds
  --diff_priv DIFF_PRIV
                        run VAE with differential privacy
```

### Dataset

Experiments are run against the [Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT) dataset](https://biostat.app.vumc.org/wiki/Main/SupportDesc) accessed via the [pycox](https://github.com/havakv/pycox) python library.

## Roadmap

See the [open issues](https://github.com/nhsx/SynthVAE/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

## License

Distributed under the MIT License. _See [LICENSE](./LICENSE) for more information._

## Contact

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [analytics-unit@nhsx.nhs.uk](mailto:analytics-unit@nhsx.nhs.uk)

## Acknowledgements


