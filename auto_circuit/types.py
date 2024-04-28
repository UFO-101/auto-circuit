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
"""
Determines how mask values are used to ablate edges.

If `None`, the mask value is used directly to interpolate between the original and
ablated values. ie. `0.0` means the original value, `1.0` means the ablated value.

If `"hard_concrete"`, the mask value parameterizes a "HardConcrete" distribution
([Louizos et al., 2017](https://arxiv.org/abs/1712.01312),
[Cao et at., 2021](https://arxiv.org/abs/2104.03514))
which is sampled to interpolate between the original and ablated values. The
HardConcrete distribution allows us to optimize a continuous variable while still
allowing the mask to take values equal to `0.0` or `1.0`. And the stochasticity helps to
reduce problems from vanishing gradients.

If `"sigmoid"`, the mask value is passed through a sigmoid function and then used to
interpolate between the original and ablated values.
"""

# Define a colorblind-friendly palette
COLOR_PALETTE = [
    "rgb(55, 126, 184)",  # blue
    "rgb(255, 127, 0)",  # orange
    "rgb(77, 175, 74)",  # green
    "rgb(247, 129, 191)",  # pink
    "rgb(228, 26, 28)",  # red
    "rgb(152, 78, 163)",  # purple
    "rgb(166, 86, 40)",  # brown
    "rgb(153, 153, 153)",  # grey
    "rgb(222, 222, 0)",  # yellow
]

# Create or modify a template
template = pio.templates["plotly"]
template.layout.colorway = COLOR_PALETTE  # type: ignore
template.layout.font.size = 19  # type: ignore

# Set the template as the default
pio.templates.default = "plotly"


class EdgeCounts(Enum):
    """Special values for [`TestEdges`][auto_circuit.types.TestEdges] that get computed
    at runtime."""

    ALL = 1
    """Test `0, 1, 2, ..., n_edges` edges."""

    LOGARITHMIC = 2
    """Test `0, 1, 2, ..., 10, 20, ..., 100, 200, ..., 1000, 2000, ...` edges."""

    GROUPS = 3
    """
    Group edges by [`PruneScores`][auto_circuit.types.PruneScores] and cumulatively add
    the number of edges in each group in descending order by score.
    """


TestEdges = EdgeCounts | List[int | float]
"""
Determines the set of [number of edges to prune] to test. This value is used as a
parameter to [`edge_counts_util`][auto_circuit.utils.graph_utils.edge_counts_util].

If a list of integers, then these are the edge counts that will be used. If a list of
floats, then these proportions of the total number of edges will be used.
"""

AutoencoderInput = Literal["mlp_post_act", "resid_delta_mlp", "resid"]
"""The activation in each layer that is replaced by an autoencoder reconstruction."""

OutputSlice = Optional[Literal["last_seq", "not_first_seq"]]
"""
The slice of the output that is considered for task evaluation. For example,
`"last_seq"` will consider the last token's output in transformer models.
"""


class PatchType(Enum):
    """Whether to patch the edges in the circuit or the complement of the circuit."""

    EDGE_PATCH = 1
    """Patch the edges in the circuit."""

    TREE_PATCH = 2
    """Patch the edges <u>not</u> in the circuit."""

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


class AblationType(Enum):
    """
    Type of activation with which replace an original activation during a forward pass.
    """

    RESAMPLE = 1
    """Use the corresponding activation from the forward pass of the corrupt input."""

    ZERO = 2
    """Use a vector of zeros."""

    TOKENWISE_MEAN_CLEAN = 3
    """Compute the token-wise mean of the clean input over the entire dataset."""

    TOKENWISE_MEAN_CORRUPT = 4
    """Compute the token-wise mean of the corrupt input over the entire dataset."""

    TOKENWISE_MEAN_CLEAN_AND_CORRUPT = 5
    """
    Compute the token-wise mean of the clean and corrupt inputs over the entire dataset.
    """

    BATCH_TOKENWISE_MEAN = 6
    """Compute the token-wise mean over the current input batch."""

    BATCH_ALL_TOK_MEAN = 7
    """Compute the mean over all tokens in the current batch."""

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()

    @property
    def mean_over_dataset(self) -> bool:
        return self in {
            AblationType.TOKENWISE_MEAN_CLEAN,
            AblationType.TOKENWISE_MEAN_CORRUPT,
            AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
        }

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
    """
    A node in the computational graph of the model used for ablation.

    Args:
        name: The name of the node.
        module_name: The name of the PyTorch module in the model that the node is in.
            Modules can have multiple nodes, for example, the multi-head attention
            module in a transformer model has a node for each head.
        layer: The layer of the model that the node is in. Transformer blocks count as 2
            layers (one for the attention layer and one for the MLP layer) because we
            want to connect nodes in the attention layer to nodes in the subsequent MLP
            layer.
        head_idx: The index of the head in the multi-head attention module that the node
            is in.
        head_dim: The dimension of the head in the multi-head attention layer that the
            node is in.
        weight: The name of the weight in the module that corresponds to the node. Not
            currently used, but could be used by a circuit finding algorithm.
        weight_head_dim: The dimension of the head in the weight tensor that corresponds
            to the node. Not currently used, but could be used by a circuit finding
            algorithm.
    """

    name: str
    module_name: str
    layer: int  # Layer of the model (transformer blocks count as 2 layers)
    head_idx: Optional[int] = None
    head_dim: Optional[int] = None
    weight: Optional[str] = None
    weight_head_dim: Optional[int] = None

    def module(self, model: Any) -> PatchWrapper:
        """
        Get the [`PatchWrapper`][auto_circuit.utils.patch_wrapper.PatchWrapper] for this
        node.

        Args:
            model: The model that the node is in.

        Returns:
            The `PatchWrapper` for this node.
        """
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


PruneScores = Dict[str, t.Tensor]
"""
Dictionary from module names of [`DestNode`s][auto_circuit.types.DestNode] to edge
scores. The edge scores are stored as a tensor where each value corresponds to the score
of an incoming [`Edge`][auto_circuit.types.Edge].
"""


@dataclass(frozen=True)
class Edge:
    """
    A directed edge from a [`SrcNode`][auto_circuit.types.SrcNode] to a
    [`DestNode`][auto_circuit.types.DestNode] in the computational graph of the model
    used for ablation.

    And an optional sequence index that specifies the token position when the
    [`PatchableModel`][auto_circuit.utils.patchable_model.PatchableModel] has `seq_len`
    not `None`.
    """

    src: SrcNode
    """The [`SrcNode`][auto_circuit.types.SrcNode] of the edge."""
    dest: DestNode
    """The [`DestNode`][auto_circuit.types.DestNode] of the edge."""
    seq_idx: Optional[int] = None
    """The sequence index of the edge."""

    @property
    def name(self) -> str:
        """The name of the edge. Equal to `{src.name}->{dest.name}`."""
        return f"{self.src.name}->{self.dest.name}"

    @property
    def patch_idx(self) -> Tuple[int, ...]:
        """The index of the edge in the `patch_mask` or
        [`PruneScores`][auto_circuit.types.PruneScores] tensor of the `dest` node."""
        seq_idx = [] if self.seq_idx is None else [self.seq_idx]
        head_idx = [] if self.dest.head_idx is None else [self.dest.head_idx]
        return tuple(seq_idx + head_idx + [self.src.src_idx - self.dest.min_src_idx])

    def patch_mask(self, model: Any) -> t.nn.Parameter:
        """The `patch_mask` tensor of the `dest` node."""
        return self.dest.module(model).patch_mask

    def prune_score(self, prune_scores: PruneScores) -> t.Tensor:
        """
        The score of the edge in the given
        [`PruneScores`][auto_circuit.types.PruneScores].
        """
        return prune_scores[self.dest.module_name][self.patch_idx]

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


PruneMetricKey = str
"""
A string the uniquely identifies a
[`PruneMetric`][auto_circuit.metrics.prune_metrics.prune_metrics.PruneMetric].
"""

TaskKey = str
"""A string that uniquely identifies a [`Task`][auto_circuit.tasks.Task]."""

AlgoKey = str
"""
A string that uniquely identifies a
[`PruneAlgo`][auto_circuit.prune_algos.prune_algos.PruneAlgo].
"""


Measurements = List[Tuple[int | float, int | float]]
"""
List of X and Y measurements. X is often the number of edges in the circuit and Y is
often some measure of faithfulness.
"""

BatchOutputs = Dict[BatchKey, t.Tensor]
"""
A dictionary mapping from [`BatchKey`s][auto_circuit.data.BatchKey] to output
tensors.
"""

CircuitOutputs = Dict[int, BatchOutputs]
"""
A dictionary mapping from the number of pruned edges to
[`BatchOutputs`][auto_circuit.types.BatchOutputs]
"""


AlgoPruneScores = Dict[AlgoKey, PruneScores]
"""
A dictionary mapping from [`AlgoKey`s][auto_circuit.types.AlgoKey] to
[`PruneScores`][auto_circuit.types.PruneScores].
"""

TaskPruneScores = Dict[TaskKey, AlgoPruneScores]
"""
A dictionary mapping from [`TaskKey`s][auto_circuit.types.TaskKey] to
[`AlgoPruneScores`][auto_circuit.types.AlgoPruneScores].
"""

AlgoMeasurements = Dict[AlgoKey, Measurements]
"""
A dictionary mapping from [`AlgoKey`s][auto_circuit.types.AlgoKey] to
[`Measurements`][auto_circuit.types.Measurements].
"""

TaskMeasurements = Dict[TaskKey, AlgoMeasurements]
"""
A dictionary mapping from [`TaskKey`s][auto_circuit.types.TaskKey] to
[`AlgoMeasurements`][auto_circuit.types.AlgoMeasurements].
"""

PruneMetricMeasurements = Dict[PruneMetricKey, TaskMeasurements]
"""
A dictionary mapping from [`PruneMetricKey`s][auto_circuit.types.PruneMetricKey] to
[`TaskMeasurements`][auto_circuit.types.TaskMeasurements].
"""

AblationMeasurements = Dict[AblationType, PruneMetricMeasurements]
"""
A dictionary mapping from [`AblationType`s][auto_circuit.types.AblationType] to
[`PruneMetricMeasurements`][auto_circuit.types.PruneMetricMeasurements].
"""
