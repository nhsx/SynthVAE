import argparse


def add_all_module_args(parser: argparse.ArgumentParser):
    dataloader_group = parser.add_argument_group(title="dataloader")
    add_dataloader_args(dataloader_group)
    structure_group = parser.add_argument_group(title="structure")
    add_structure_args(structure_group)
    model_group = parser.add_argument_group(title="model")
    add_model_args(model_group)
    evaluation_group = parser.add_argument_group(title="evaluation")
    add_evaluation_args(evaluation_group)
    plotting_group = parser.add_argument_group(title="plotting")
    add_plotting_args(plotting_group)


def add_config_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input-config",
        "-c",
        required=True,
        help="Specify the config file to unpack.",
    )
    overrides_group = parser.add_argument_group(title="overrides")
    # TODO is there a way to do this using `add_all_module_args`, i.e. can we nest groups? Doesn't seem to work
    add_dataloader_args(overrides_group, override=True)
    add_structure_args(overrides_group, override=True)
    add_model_args(overrides_group, override=True)
    add_evaluation_args(overrides_group, override=True)
    add_plotting_args(overrides_group, override=True)


def add_dataloader_args(parser: argparse.ArgumentParser, override=False):
    parser.add_argument(
        "--input-file",
        "-i",
        required=(not override),
        help="Specify the name of the `.csv` file to prepare.",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default="_prepared",
        help="Specify where to write the prepared data, defaults to `\{args.dir\}/\{args.input_file\}_prepared.csv`.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="data",
        help="Specify the directory to read and write data from and to, defaults to `./data`.",
    )
    parser.add_argument(
        "--imputation-strategy",
        "--impute-strategy",
        "--impute",
        "-is",
        default="mean",
        choices=["mean", "median", "cull"],
        help="Specify the imputation strategy for missing values, defaults to inserting the mean of the relevant column.",
    )


def add_structure_args(parser: argparse.ArgumentParser, override=False):
    pass


def add_model_args(parser: argparse.ArgumentParser, override=False):
    pass


def add_evaluation_args(parser: argparse.ArgumentParser, override=False):
    pass


def add_plotting_args(parser: argparse.ArgumentParser, override=False):
    pass
