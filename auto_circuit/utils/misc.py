from contextlib import contextmanager
from functools import reduce
from typing import Iterator, Set

import torch as t
from torch.utils.hooks import RemovableHandle


@contextmanager
def remove_hooks() -> Iterator[Set[RemovableHandle]]:
    handles: Set[RemovableHandle] = set()
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def module_by_name(model: t.nn.Module, module_name: str) -> t.nn.Module:
    return reduce(getattr, [model] + module_name.split("."))  # type: ignore


def set_module_by_name(
    model: t.nn.Module, module_name: str, new_module: t.nn.Module
) -> None:
    parent = model
    if "." in module_name:
        parent = reduce(getattr, [model] + module_name.split(".")[:-1])  # type: ignore
    setattr(parent, module_name.split(".")[-1], new_module)


def percent_gpu_mem_used(total_gpu_mib: int = 49000) -> str:
    return (
        "Memory used {:.1f}".format(
            ((t.cuda.memory_allocated() / (2**20)) / total_gpu_mib) * 100
        )
        + "%"
    )
