from functools import partial

from tqdm.auto import tqdm

tqdm = partial(
    tqdm,
    dynamic_ncols=True,
    bar_format="{desc}{bar}{r_bar}",
    leave=None,
    # colour="#03CF0A",
    delay=0,
)
"""
Wrapper around `tqdm` with default settings for the project. Note that the `tqdm`
dependency in this repo is my [fork](https://github.com/UFO-101/tqdm) that fixes a
[rendering issue in Jupyter notebooks](https://github.com/microsoft/vscode-jupyter/issues/9397).
I've made a [pull request](https://github.com/tqdm/tqdm/pull/1504) to the original repo
but it hasn't been merged yet.
"""
