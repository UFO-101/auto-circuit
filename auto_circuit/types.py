from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.utils.misc import module_by_name
from auto_circuit.utils.patch_wrapper import PatchWrapper


class EdgeCounts(Enum):
    ALL = 1
    LOGARITHMIC = 2
    GROUPS = 3  # Group edges by score and add each edge in the group at the same time.


TestEdges = EdgeCounts | List[int | float]


class PatchType(Enum):
    EDGE_PATCH = 1
    TREE_PATCH = 2

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


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
    seq_idx: Optional[int] = None

    @property
    def name(self) -> str:
        return f"{self.src.name}->{self.dest.name}"

    @property
    def patch_idx(self) -> Tuple[int, ...]:
        seq_idx = [] if self.seq_idx is None else [self.seq_idx]
        head_idx = [] if self.dest.head_idx is None else [self.dest.head_idx]
        return tuple(seq_idx + head_idx + [self.src.idx])

    def patch_mask(self, model: t.nn.Module) -> t.nn.Parameter:
        return self.dest.module(model).patch_mask

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Task:
    name: str
    model: t.nn.Module
    train_loader: PromptDataLoader
    test_loader: PromptDataLoader
    true_edge_func: Callable[..., Set[Edge]]
    token_circuit: bool = False

    @property
    def true_edges(self) -> Set[Edge]:
        if self.token_circuit:
            return self.true_edge_func(self.model, True)
        else:
            return self.true_edge_func(self.model)


Measurements = List[Tuple[int | float, int | float]]
PrunedOutputs = Dict[int, List[t.Tensor]]
PruneScores = Dict[Edge, float]


@dataclass(frozen=True)
class PruneAlgo:
    name: str
    func: Callable[[Task], PruneScores]
    short_name: Optional[str] = None

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PruneAlgo):
            return False
        return self.name == __value.name and self.func == __value.func


@dataclass(frozen=True)
class Metric:
    name: str
    metric_func: Callable[
        [Task, Optional[PruneScores], Optional[PrunedOutputs]], Measurements
    ]
    log_x: bool = False
    log_y: bool = False
    lower_better: bool = False
    y_axes_match: bool = False  # Whether to use the same y-axis for all tasks
    y_min: Optional[float] = None

    def _post_init(self) -> None:
        if self.log_y:
            assert self.y_min is not None


AlgoPruneScores = Dict[PruneAlgo, PruneScores]
TaskPruneScores = Dict[Task, AlgoPruneScores]

AlgoMeasurements = Dict[PruneAlgo, Measurements]
TaskMeasurements = Dict[Task, AlgoMeasurements]
MetricMeasurements = Dict[Metric, TaskMeasurements]
