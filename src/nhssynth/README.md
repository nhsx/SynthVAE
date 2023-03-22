# Modules

This folder contains all of the modules contained in this package. They can be used together or independently - through importing them into your existing codebase or using the `cli` module and `runner.py` to select which / all modules to run.

## Importing a module from this package

After installing the package, you can simply do:
```python
from nhssynth import <module>
```
and you will be able to use it in your code!

## Creating a new module and folding it into the CLI

The following instructions specify how to extend this package with a new module:

1. Create a folder for your module within the package, i.e. `src/nhssynth/mymodule`
2. Include within it a main executor that accepts arguments from the `cli` module, e.g. `def myexecutor(args): ...` in `mymodule/executor.py` and export this by adding `from .executor import myexecutor` in `mymodule/__init__.py`.
3. In the `cli` module folder, add the following code blocks to `run.py` (the second is optional depending on whether this module should be executed as part of a full pipeline run):
    ```python
    from modules import ..., mymodule, ...

    ...

    def run()
        ...
        parser_mymodule = subparsers.add_parser(
            name="mymodule",
            description=...,
            help=...,
        )
        add_mymodule_args(parser_mymodule)
        parser_mymodule.set_defaults(func=mymodule.executor)
        ...
    ```
    ```python
    def run_pipeline(args):
        ...
        mymodule.executor(args)
        ...
    ```
4. Similarly, add the following code blocks to `arguments.py` (again, the second block is optional):
    ```python
    def add_mymodule_args(parser: argparse.ArgumentParser):
        ...
    ```
    ```python
    def add_all_module_args(parser: argparse.ArgumentParser):
        ...
        mymodule_group = parser.add_argument_group(title="mymodule")
        add_mymodule_args(mymodule_group)
        ...

    ...

    def add_mymodule_args(parser: argparse.ArgumentParser, override=False):
        ...
        add_mymodule_args(overrides_group)
        ...
    ```
5. After populating the functions in a similar fashion to the existing modules, your module will work as part of the CLI!

