from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch as t


class ActType(Enum):
    """Type of activation. Used to determine network inputs and patch values."""

    CLEAN = 1
    CORRUPT = 2
    ZERO = 3
    # MEAN = 4

    def __str__(self) -> str:
        return self.name[0] + self.name[1:].lower()


class EdgeCounts(Enum):
    ALL = 1
    LOGARITHMIC = 2


@dataclass(frozen=True)
class ExperimentType:
    input_type: ActType
    patch_type: ActType
    sort_prune_scores_high_to_low: bool = True


HashableTensorIndex = Tuple[Optional[int], ...] | None | int
TensorIndex = Tuple[int | slice, ...] | int | slice


def tensor_index_to_slice(t_idx: HashableTensorIndex) -> TensorIndex:
    if t_idx is None:
        return slice(None)
    elif isinstance(t_idx, int):
        return t_idx
    return tuple([slice(None) if idx is None else idx for idx in t_idx])


@dataclass(frozen=True)
class EdgeSrc:
    name: str
    module: t.nn.Module
    _t_idx: HashableTensorIndex
    weight: str
    _weight_t_idx: HashableTensorIndex

    @property
    def t_idx(self) -> TensorIndex:
        return tensor_index_to_slice(self._t_idx)

    @property
    def weight_t_idx(self) -> TensorIndex:
        return tensor_index_to_slice(self._weight_t_idx)


@dataclass(frozen=True)
class EdgeDest:
    name: str
    module: t.nn.Module
    kwarg: Optional[str]
    _t_idx: HashableTensorIndex
    weight: str
    _weight_t_idx: HashableTensorIndex

    @property
    def t_idx(self) -> TensorIndex:
        return tensor_index_to_slice(self._t_idx)

    @property
    def weight_t_idx(self) -> TensorIndex:
        return tensor_index_to_slice(self._weight_t_idx)


@dataclass(frozen=True)
class Edge:
    src: EdgeSrc
    dest: EdgeDest

    def __repr__(self) -> str:
        return f"{self.src.name}->{self.dest.name}"

    def __str__(self) -> str:
        return self.__repr__()
