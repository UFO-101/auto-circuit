from typing import Any, Dict, List, Optional, Set, Tuple

import torch as t
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

from auto_circuit.types import DestNode, Edge, Node, PruneScores, SrcNode
from auto_circuit.utils.patch_wrapper import PatchWrapperImpl


class PatchableModel(t.nn.Module):
    """
    A model that can be ablated along individual edges in its computation graph.

    This class has many of the same methods and attributes as TransformerLens'
    `HookedTransformer`s. These are simple wrappers which pass through to the
    implementation in the wrapped model. These methods and attributes
    are:
    <ul>
        <li><code>forward</code></li>
        <li><code>run_with_cache</code></li>
        <li><code>add_hook</code></li>
        <li><code>reset_hooks</code></li>
        <li><code>cfg</code></li>
        <li><code>tokenizer</code></li>
        <li><code>input_to_embed</code></li>
        <li><code>blocks</code></li>
        <li><code>to_tokens</code></li>
        <li><code>to_str_tokens</code></li>
        <li><code>to_string</code></li>
    </ul>

    Args:
        nodes: The set of all nodes in the computation graph.
        srcs: The set of all source nodes in the computation graph.
        dests: The set of all destination nodes in the computation graph.
        edge_dict: A dictionary mapping sequence positions to the edges at that
            position.
        edges: The set of all edges in the computation graph.
        seq_dim: The sequence dimension of the model. This is the dimension on which new
            inputs are concatenated. In transformers, this is `1` because the
            activations are of shape `[batch_size, seq_len, hidden_dim]`.
        seq_len: The sequence length of the model inputs. If `None`, all token positions
            are simultaneously ablated.
        wrappers: The set of all `PatchWrapper`s in the model.
        src_wrappers: The set of all `PatchWrapper`s that are source nodes.
        dest_wrappers: The set of all `PatchWrapper`s that are destination nodes.
        out_slice: Specifies the index/slice of the output of the model to be
            considered for the task.
        is_factorized: Whether the model is factorized, for Edge Ablation. Otherwise,
            only Node Ablation is possible.
        is_transformer: Whether the model is a transformer.
        separate_qkv: Whether the model has separate query, key, and value inputs. Only
            used for transformers.
        kv_caches: A dictionary mapping batch sizes to the past key-value caches of the
            transformer. Only used for transformers.
        wrapped_model: The model to wrap which is being made patchable.
    """

    nodes: Set[Node]
    srcs: Set[SrcNode]
    dests: Set[DestNode]
    edge_dict: Dict[int | None, List[Edge]]  # Key is token position or None for all
    edges: Set[Edge]
    n_edges: int
    seq_dim: int
    seq_len: Optional[int]
    wrappers: Set[PatchWrapperImpl]
    src_wrappers: Set[PatchWrapperImpl]
    dest_wrappers: Set[PatchWrapperImpl]
    patch_masks: Dict[str, t.nn.Parameter]
    out_slice: Tuple[slice | int, ...]
    is_factorized: bool
    is_transformer: bool
    separate_qkv: Optional[bool]
    kv_caches: Optional[Dict[int, HookedTransformerKeyValueCache]]
    wrapped_model: t.nn.Module

    def __init__(
        self,
        nodes: Set[Node],
        srcs: Set[SrcNode],
        dests: Set[DestNode],
        edge_dict: Dict[int | None, List[Edge]],
        edges: Set[Edge],
        seq_dim: int,
        seq_len: Optional[int],
        wrappers: Set[PatchWrapperImpl],
        src_wrappers: Set[PatchWrapperImpl],
        dest_wrappers: Set[PatchWrapperImpl],
        out_slice: Tuple[slice | int, ...],
        is_factorized: bool,
        is_transformer: bool,
        separate_qkv: Optional[bool],
        kv_caches: Tuple[Optional[HookedTransformerKeyValueCache], ...],
        wrapped_model: t.nn.Module,
    ) -> None:
        super().__init__()
        self.nodes = nodes
        self.srcs = srcs
        self.dests = dests
        self.edge_dict = edge_dict
        self.edges = edges
        self.n_edges = len(edges)
        self.seq_dim = seq_dim
        self.seq_len = seq_len
        self.wrappers = wrappers
        self.src_wrappers = src_wrappers
        self.dest_wrappers = dest_wrappers
        self.patch_masks = {}
        for dest_wrapper in self.dest_wrappers:
            self.patch_masks[dest_wrapper.module_name] = dest_wrapper.patch_mask
        self.out_slice = out_slice
        self.is_factorized = is_factorized
        self.is_transformer = is_transformer
        if is_transformer:
            assert separate_qkv is not None
        self.separate_qkv = separate_qkv
        if all([kv_cache is None for kv_cache in kv_caches]) or len(kv_caches) == 0:
            self.kv_caches = None
        else:
            self.kv_caches = {}
            for kv_cache in kv_caches:
                if kv_cache is not None:
                    batch_size = kv_cache.previous_attention_mask.shape[0]
                    self.kv_caches[batch_size] = kv_cache
        self.wrapped_model = wrapped_model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Wrapper around the forward method of the wrapped model. If `kv_caches` is not
        `None`, the KV cache is passed to the wrapped model as a keyword argument.
        """
        if self.kv_caches is None or "past_kv_cache" in kwargs:
            return self.wrapped_model(*args, **kwargs)
        else:
            batch_size = args[0].shape[0]
            kv = self.kv_caches[batch_size]
            return self.wrapped_model(*args, past_kv_cache=kv, **kwargs)

    def run_with_cache(self, *args: Any, **kwargs: Any) -> Any:
        """
        Wrapper around the `run_with_cache` method of the wrapped TransformerLens
        `HookedTransformer`. If `kv_caches` is not `None`, the KV cache is passed to the
        wrapped model as a keyword argument.
        """
        if self.kv_caches is None:
            return self.wrapped_model.run_with_cache(*args, **kwargs)
        else:
            batch_size = args[0].shape[0]
            kv = self.kv_caches[batch_size]
            return self.wrapped_model.run_with_cache(*args, past_kv_cache=kv, **kwargs)

    def new_prune_scores(self, init_val: float = 0.0) -> PruneScores:
        """
        A new [`PruneScores`][auto_circuit.types.PruneScores] instance with the same
        keys and shapes as the current patch masks, initialized to `init_val`.

        Args:
            init_val: The initial value to set all the prune scores to.

        Returns:
            A new [`PruneScores`][auto_circuit.types.PruneScores] instance.
        """
        prune_scores: PruneScores = {}
        for (mod_name, mask) in self.patch_masks.items():
            prune_scores[mod_name] = t.full_like(mask.data, init_val)
        return prune_scores

    def circuit_prune_scores(self, edges: Set[Edge], bool: bool = False) -> PruneScores:
        """
        Convert a set of edges to a corresponding
        [`PruneScores`][auto_circuit.types.PruneScores] object.

        Args:
            edges: The set of edges to convert to prune scores.
            bool: Whether to return the prune scores as boolean type tensors.

        Returns:
            The prune scores corresponding to the set of edges.
        """
        prune_scores = self.new_prune_scores()
        for edge in edges:
            prune_scores[edge.dest.module_name][edge.patch_idx] = 1.0
        if bool:
            return dict([(mod, mask.bool()) for (mod, mask) in prune_scores.items()])
        else:
            return prune_scores

    def current_patch_masks_as_prune_scores(self) -> PruneScores:
        """
        Convert the current patch masks to a corresponding
        [`PruneScores`][auto_circuit.types.PruneScores] object.

        Returns:
            The prune scores corresponding to the current patch masks.
        """
        return dict([(mod, mask.data) for (mod, mask) in self.patch_masks.items()])

    def add_hook(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.add_hook(*args, **kwargs)

    def reset_hooks(self) -> Any:
        return self.wrapped_model.reset_hooks()

    @property
    def cfg(self) -> Any:
        return self.wrapped_model.cfg

    @property
    def tokenizer(self) -> Any:
        return self.wrapped_model.tokenizer

    @property
    def input_to_embed(self) -> Any:
        return self.wrapped_model.input_to_embed

    @property
    def blocks(self) -> Any:
        return self.wrapped_model.blocks

    def to_tokens(self) -> Any:
        return self.wrapped_model.to_tokens

    def to_str_tokens(self) -> Any:
        return self.wrapped_model.to_str_tokens

    def to_string(self, *args: Any, **kwargs: Any) -> Any:
        return self.wrapped_model.to_string(*args, **kwargs)

    def __str__(self) -> str:
        return self.wrapped_model.__str__()

    def __repr__(self) -> str:
        return self.wrapped_model.__repr__()
