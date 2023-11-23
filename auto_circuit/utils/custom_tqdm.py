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
