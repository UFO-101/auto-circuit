from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch as t

from auto_circuit.utils.misc import module_by_name
from auto_circuit.utils.patch_wrapper import PatchWrapper


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


TestEdges = EdgeCounts | List[int | float]


@dataclass(frozen=True)
class ExperimentType:
    input_type: ActType
    patch_type: ActType
    decrease_prune_scores: bool = True


@dataclass(frozen=True)
class Node:
    name: str
    module_name: str
    layer: int  # Layer of the model (transformer blocks count as 2 layers)
    idx: int = 0  # Index of the node across all src/dest nodes in all layers
    head_idx: Optional[int] = None
    head_dim: Optional[int] = None
    weight: Optional[str] = None
    weight_head_dim: Optional[int] = None

    def module(self, model: t.nn.Module) -> PatchWrapper:
        patch_wrapper = module_by_name(model, self.module_name)
        assert isinstance(patch_wrapper, PatchWrapper)
        return patch_wrapper

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class SrcNode(Node):
    """A node that is the source of an edge."""


class DestNode(Node):
    """A node that is the destination of an edge."""


@dataclass(frozen=True)
class Edge:
    src: SrcNode
    dest: DestNode

    @property
    def name(self) -> str:
        return f"{self.src.name}->{self.dest.name}"

    @property
    def patch_idx(self) -> Tuple[int, ...]:
        dest_idx = [] if self.dest.head_idx is None else [self.dest.head_idx]
        return tuple(dest_idx + [self.src.idx])

    def patch_mask(self, model: t.nn.Module) -> t.nn.Parameter:
        return self.dest.module(model).patch_mask

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name
