import argparse

import yaml
from nhssynth.modules import dataloader, evaluation, model, plotting, structure

from .arguments import *
from .config import *


def run_pipeline(args):
    print("Running full pipeline...")
    dataloader.run(args)
    structure.run(args)
    model.run(args)
    evaluation.run(args)
    plotting.run(args)


def run():

    parser = argparse.ArgumentParser(
        prog="SynthVAE", description="CLI for preparing, training and evaluating a synthetic data generator."
    )

    # Below we instantiate one subparser for each module + one for running with config file and one for doing a full run with CLI-specified config
    subparsers = parser.add_subparsers()

    parser_full = subparsers.add_parser(
        name="pipeline",
        description="Run an automatically configured module or set of modules specified by a tape file in `tapes/`. Note that you can override parts of the configuration on the fly by using the usual CLI flags.",
        help="Run full pipeline",
    )
    add_all_module_args(parser_full)
    parser_full.set_defaults(func=run_pipeline)

    parser_config = subparsers.add_parser(
        name="config",
        description="Run module(s) according to configuration specified by a tape file in `tapes/`. Note that you can override parts of the configuration on the fly by using the usual CLI flags.",
        help="Run module(s) using configuration tape file",
    )
    add_config_args(parser_config)
    parser_config.set_defaults(func=read_config)

    parser_dataloader = subparsers.add_parser(
        name="prepare",
        description="Run the Data Loader module, to prepare data for use in other modules.",
        help="Prepare input data",
    )
    add_dataloader_args(parser_dataloader)
    parser_dataloader.set_defaults(func=dataloader.run)

    parser_structure = subparsers.add_parser(
        name="structure",
        description="Run the Structural Discovery module, to learn a structural model for use in training and evaluation.",
        help="Discover structure",
    )
    add_structure_args(parser_structure)
    parser_structure.set_defaults(func=structure.run)

    parser_model = subparsers.add_parser(
        name="train",
        description="Run the Architecture module, to train a model.",
        help="Train a model",
    )
    add_model_args(parser_model)
    parser_model.set_defaults(func=model.run)

    parser_evaluation = subparsers.add_parser(
        name="evaluate",
        description="Run the Evaluation module, to evaluate a model.",
        help="Evaluate a model",
    )
    add_evaluation_args(parser_evaluation)
    parser_evaluation.set_defaults(func=evaluation.run)

    parser_plotting = subparsers.add_parser(
        name="plot",
        description="Run the Evaluation module, to generate plots for a given model and / or evaluation.",
        help="Generate plots",
    )
    add_plotting_args(parser_plotting)
    parser_plotting.set_defaults(func=plotting.run)

    args = parser.parse_args()
    # TODO come up with a better solution than try:catch, perhaps it is possible to check if a subparser default func has been set
    try:
        # Run the appropriate function depending on the positional option selected
        args.func(args)
    except:
        parser.parse_args(["--help"])

    print(yaml.dump(vars(args)))

    print("Complete!")


if __name__ == "__main__":
    run()
