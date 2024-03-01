from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

import plotly.io as pio
import torch as t

from auto_circuit.data import BatchKey
from auto_circuit.utils.misc import module_by_name


class PatchWrapper(t.nn.Module, ABC):
    """Abstract class for a wrapper around a module that can be patched."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any):
        pass


MaskFn = Optional[Literal["hard_concrete", "sigmoid"]]

# Define a colorblind-friendly palette
COLOR_PALETTE = [
    "#377eb8",  # blue
    "#ff7f00",  # orange
    "#4daf4a",  # green
    "#f781bf",  # pink
    "#e41a1c",  # red
    "#984ea3",  # purple
    "#a65628",  # brown
    "#999999",  # grey
    "#dede00",  # yellow
]

# Create or modify a template
template = pio.templates["plotly"]
template.layout.colorway = COLOR_PALETTE  # type: ignore
template.layout.font.size = 19  # type: ignore

# Set the template as the default
pio.templates.default = "plotly"


class EdgeCounts(Enum):
    ALL = 1
    LOGARITHMIC = 2
    GROUPS = 3  # Group edges by score and add each edge in the group at the same time.


TestEdges = EdgeCounts | List[int | float]
AutoencoderInput = Literal["mlp_post_act", "resid_delta_mlp", "resid"]
OutputSlice = Optional[Literal["last_seq", "not_first_seq"]]


class PatchType(Enum):
    EDGE_PATCH = 1
    TREE_PATCH = 2

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


class AblationType(Enum):
    RESAMPLE = 1
    ZERO = 2
    TOKENWISE_MEAN_CLEAN = 3
    TOKENWISE_MEAN_CORRUPT = 4
    TOKENWISE_MEAN_CLEAN_AND_CORRUPT = 5

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()

    @property
    def mean_over_dataset(self) -> bool:
        return self not in {AblationType.RESAMPLE, AblationType.ZERO}

    @property
    def clean_dataset(self) -> bool:
        return self in {
            AblationType.TOKENWISE_MEAN_CLEAN,
            AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
        }

    @property
    def corrupt_dataset(self) -> bool:
        return self in {
            AblationType.TOKENWISE_MEAN_CORRUPT,
            AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
        }


@dataclass(frozen=True)
class Node:
    name: str
    module_name: str
    layer: int  # Layer of the model (transformer blocks count as 2 layers)
    head_idx: Optional[int] = None
    head_dim: Optional[int] = None
    weight: Optional[str] = None
    weight_head_dim: Optional[int] = None

    def module(self, model: Any) -> PatchWrapper:
        patch_wrapper = module_by_name(model, self.module_name)
        assert isinstance(patch_wrapper, PatchWrapper)
        return patch_wrapper

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class SrcNode(Node):
    """A node that is the source of an edge."""

    src_idx: int = 0  # Index of the node across all src nodes in all layers


@dataclass(frozen=True)
class DestNode(Node):
    """A node that is the destination of an edge."""

    min_src_idx: int = 0  # min src_idx of all incoming SrcNodes (0 in factorized model)


PruneScores = Dict[str, t.Tensor]  # module_name -> edge scores


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
        return tuple(seq_idx + head_idx + [self.src.src_idx - self.dest.min_src_idx])

    def patch_mask(self, model: Any) -> t.nn.Parameter:
        return self.dest.module(model).patch_mask

    def prune_score(self, prune_scores: PruneScores) -> t.Tensor:
        return prune_scores[self.dest.module_name][self.patch_idx]

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


PruneMetricKey = str
TaskKey = str
AlgoKey = str


Measurements = List[Tuple[int | float, int | float]]
BatchOutputs = Dict[BatchKey, t.Tensor]
CircuitOutputs = Dict[int, BatchOutputs]


AlgoPruneScores = Dict[AlgoKey, PruneScores]
TaskPruneScores = Dict[TaskKey, AlgoPruneScores]

AlgoMeasurements = Dict[AlgoKey, Measurements]
TaskMeasurements = Dict[TaskKey, AlgoMeasurements]
PruneMetricMeasurements = Dict[PruneMetricKey, TaskMeasurements]
AblationMeasurements = Dict[AblationType, PruneMetricMeasurements]
