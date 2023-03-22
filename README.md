# NHS Synth

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8%20--%203.10-blue)](https://www.python.org/downloads/release/python-31010/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat)](https://pycqa.github.io/isort/)

</div>

## About the Project

The project currently consists of a Python package alongside research and investigative materials covering the effectiveness of the package and synthetic data more generally when applied to NHS use cases.

[Project Description - Synthetic Data Exploration: Variational Autoencoders](https://nhsx.github.io/nhsx-internship-projects/synthetic-data-exploration-vae/)

The codebase builds on previous NHSX Analytics Unit PhD internships contextualising and investigating the potential use of Variational Auto Encoders (VAEs) for synthetic data generation. These were undertaken by Dominic Danks ([last commit to the repository: 88a4bdf](https://github.com/nhsx/NHSSynth/commit/88a4bdf613f538af45834f22d38e52312cfe24c5)) and David Brind ([last commit to the repository: ]()).

_**Note:** No data, public or private are shared in this repository._

## Getting Started

### Project Stucture

- The main package and codebase is found in [`src/nhssynth`]() (see Usage below for more information)
- Accompanying materials are available in the `docs` folder:
  - A [report](docs/reports/report.pdf) summarising the previous iteration of this project
  - A [model card](docs/model_card.md) providing more information about the VAE with Differential Privacy
- Numerous [exemplar configurations](config) are found in `config`
- Empty `data` and `experiments` folders are provided; these are the default locations for inputs and outputs when running the project using the provided [`cli`](src/nhssynth/cli/) module
- Pre-processing notebooks for specific datasets used to assess the approach and other non-core code can be found in [`auxiliary`](auxiliary/)

### Installation

As it stands, we recommend the following steps to reproduce our experiments and fully work with this project:

1. Clone the repo
2. Ensure one of the required versions of Python is installed
3. Install [`poetry`](https://python-poetry.org/docs/#installation)
4. Instantiate a virtual environment, e.g. via `python -m venv nhssynth`
3. Activate the virtual environment, e.g. via `source nhssynth/bin/activate`
4. Install project dependencies with `poetry install` (optionally install `jupyter` and `notebook` to work with some of the preprocessing files in [`auxiliary`](auxiliary/))
5. Interact with the package in one of two ways:
    - Via the [`cli`](src/nhssynth/cli/) module using `poetry run cli`
    - Through building the package with `poetry build` and using it in an existing project (`import nhssynth`). However, if you intend on doing the latter it may be preferable to instead follow the second, simpler setup below.

For more standard usage of the package:

1. Run `pip install nhssynth` within a supported Python installation
2. Use the modules exported by the package as you would any other. _Note that in this setup you will have to work more closely with the configuration and code to ensure you are handling inputs and outputs for each module appropriately. The cli handles a lot of this complexity, and interacting with the modules directly is considered advanced usage._

### Usage

This package comprises a pipeline that is runnable via `poetry run cli pipeline <args>` or `poetry run cli config <config filepath>`. You can run the modules that make up this pipeline independently via `poetry run cli <module name>`. To see the modules that are available and their corresponding arguments and function, run `poetry run cli --help` / `poetry run cli <module name> --help`.

### Roadmap

See the [open issues](https://github.com/nhsx/NHSSynth/issues) for a list of proposed features (and known issues).

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the project
2. Create your branch (`git checkout -b <yourusername>/<featurename>`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin <yourusername>/<featurename>`)
5. Open a PR and we will try to get it merged!

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Distributed under the MIT License. _See [LICENSE](./LICENSE) for more information._

### Contact

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [analytics-unit@nhsx.nhs.uk](mailto:analytics-unit@nhsx.nhs.uk).

<!-- ### Acknowledgements -->
