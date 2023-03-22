import time
import warnings
from pathlib import Path


def check_ending(fn: str, ending=".csv") -> str:
    return fn if fn.endswith(ending) else fn + ending


def format_io(
    fn_in: str,
    fn_out: str,
    dir_data: str,
    dir_exp="experiments",
) -> tuple[Path, Path, Path]:
    # ensure .csv ending consistency
    fn_in, fn_out = check_ending(fn_in), check_ending(fn_out)

    dir_data = Path(dir_data)

    if "/" in fn_in:
        fn_in = Path(fn_in).name
        warnings.warn(
            f"Paths are not supported via `--input-file`, using the name part appended to `--dir` instead, i.e. reading from {dir_data / fn_in}"
        )

    # check if `fn_out` is given as a suffix (starts with an underscore) to append to `fn_in`, if not assume it is a name in its own right
    if fn_out[0] == "_":
        fn_out = check_ending(fn_in[:-4] + fn_out)
    else:
        fn_out

    # generate timestamped experiment folder
    dir_exp = Path(dir_exp) / time.strftime("%Y_%m_%d___%H_%M_%S")

    if "/" in fn_out:
        fn_out = Path(fn_out).name
        warnings.warn(
            f"Paths are not supported via `--output-file`, using the name part instead, i.e. writing to {dir_exp / fn_out}"
        )

    return dir_data / fn_in, dir_exp / fn_out, dir_exp
