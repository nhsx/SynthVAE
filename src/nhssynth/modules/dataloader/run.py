from .utils import *


def run(args) -> None:
    print("Preparing data")
    input_path, output_path, experiment_path = format_io(args.input_file, args.output_file, args.dir)
    print(input_path, output_path, experiment_path)
    print(args)
