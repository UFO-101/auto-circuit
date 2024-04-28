#%%
import math
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain, product
from typing import Collection, Dict, Iterator, List, Optional, Set, Tuple

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache

import auto_circuit.model_utils.micro_model_utils as mm_utils
import auto_circuit.model_utils.sparse_autoencoders.autoencoder_transformer as sae_utils
import auto_circuit.model_utils.transformer_lens_utils as tl_utils
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.model_utils.sparse_autoencoders.autoencoder_transformer import (
    AutoencoderTransformer,
)
from auto_circuit.types import (
    DestNode,
    Edge,
    EdgeCounts,
    MaskFn,
    Node,
    OutputSlice,
    PruneScores,
    SrcNode,
    TestEdges,
)
from auto_circuit.utils.misc import module_by_name, set_module_by_name
from auto_circuit.utils.patch_wrapper import PatchWrapperImpl
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import desc_prune_scores


def patchable_model(
    model: t.nn.Module,
    factorized: bool,
    slice_output: OutputSlice = None,
    seq_len: Optional[int] = None,
    separate_qkv: Optional[bool] = None,
    kv_caches: Tuple[Optional[HookedTransformerKeyValueCache], ...] = (None,),
    device: t.device = t.device("cpu"),
) -> PatchableModel:
    """
    Wrap a model and inject [`PatchWrapper`][auto_circuit.types.PatchWrapper]s into the
    node modules to enable patching.

    Args:
        model: The model to make patchable.
        factorized: Whether the model is factorized, for Edge Ablation. Otherwise,
            only Node Ablation is possible.
        slice_output: Specifies the index/slice of the output of the model to be
            considered for the task. For example, `"last_seq"` will consider the last
            token's output in transformer models.
        seq_len: The sequence length of the model inputs. If `None`, all token positions
            are simultaneously ablated.
        separate_qkv: Whether the model has separate query, key, and value inputs. Only
            used for transformers.
        kv_caches: The key and value caches for the transformer. Only used for
            transformers.
        device: The device that the model is on.

    Returns:
        The patchable model.

    Warning:
        This function modifies the model, it does not return a new model.
    """
    assert not isinstance(model, PatchableModel), "Model is already patchable"
    nodes, srcs, dests, edge_dict, edges, seq_dim, seq_len = graph_edges(
        model, factorized, separate_qkv, seq_len
    )
    wrappers, src_wrappers, dest_wrappers = make_model_patchable(
        model, factorized, srcs, nodes, device, seq_len, seq_dim
    )
    if slice_output is None:
        out_slice: Tuple[slice | int, ...] = (slice(None),)
    else:
        last_slice = [-1] if slice_output == "last_seq" else [slice(1, None)]
        out_slice: Tuple[slice | int, ...] = tuple([slice(None)] * seq_dim + last_slice)
    is_tl_transformer = isinstance(model, HookedTransformer)
    is_autoencoder_transformer = isinstance(model, AutoencoderTransformer)
    is_transformer = is_tl_transformer or is_autoencoder_transformer
    return PatchableModel(
        nodes=nodes,
        srcs=srcs,
        dests=dests,
        edge_dict=edge_dict,
        edges=edges,
        seq_dim=seq_dim,
        seq_len=seq_len,
        wrappers=wrappers,
        src_wrappers=src_wrappers,
        dest_wrappers=dest_wrappers,
        out_slice=out_slice,
        is_factorized=factorized,
        is_transformer=is_transformer,
        separate_qkv=separate_qkv,
        kv_caches=kv_caches,
        wrapped_model=model,
    )


def graph_edges(
    model: t.nn.Module,
    factorized: bool,
    separate_qkv: Optional[bool] = None,
    seq_len: Optional[int] = None,
) -> Tuple[
    Set[Node],
    Set[SrcNode],
    Set[DestNode],
    Dict[int | None, List[Edge]],
    Set[Edge],
    int,
    Optional[int],
]:
    """
    Get the nodes and edges of the computation graph of the model used for ablation.

    Args:
        model: The model to get the edges for.
        factorized: Whether the model is factorized, for Edge Ablation. Otherwise,
            only Node Ablation is possible.
        separate_qkv: Whether the model has separate query, key, and value inputs. Only
            used for transformers.
        seq_len: The sequence length of the model inputs. If `None`, all token positions
            are simultaneously ablated.

    Returns:
        Tuple containing:
            <ol>
                <li>The set of all nodes in the model.</li>
                <li>The set of all source nodes in the model.</li>
                <li>The set of all destination nodes in the model.</li>
                <li>A dictionary mapping sequence positions to the edges at that
                    position.</li>
                <li>The set of all edges in the model.</li>
                <li>The sequence dimension of the model. This is the dimension on which
                    new inputs are concatenated. For transformers, this is
                    <code>1</code> because the activations are of shape
                    <code>[batch_size, seq_len, hidden_dim]</code>.
                <li>The sequence length of the model inputs.</li>
            </ol>
    """
    seq_dim = 1
    edge_dict: Dict[Optional[int], List[Edge]] = defaultdict(list)
    if not factorized:
        if isinstance(model, MicroModel):
            srcs, dests = mm_utils.simple_graph_nodes(model)
        elif isinstance(model, HookedTransformer):
            srcs, dests = tl_utils.simple_graph_nodes(model)
        else:
            raise NotImplementedError(model)
        for i in [None] if seq_len is None else range(seq_len):
            pairs = product(srcs, dests)
            edge_dict[i] = [Edge(s, d, i) for s, d in pairs if s.layer + 1 == d.layer]
    else:
        if isinstance(model, MicroModel):
            srcs: Set[SrcNode] = mm_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = mm_utils.factorized_dest_nodes(model)
        elif isinstance(model, HookedTransformer):
            assert separate_qkv is not None, "separate_qkv must be specified for LLM"
            srcs: Set[SrcNode] = tl_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = tl_utils.factorized_dest_nodes(model, separate_qkv)
        elif isinstance(model, AutoencoderTransformer):
            assert separate_qkv is not None, "separate_qkv must be specified for LLM"
            srcs: Set[SrcNode] = sae_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = sae_utils.factorized_dest_nodes(model, separate_qkv)
        else:
            raise NotImplementedError(model)
        for i in [None] if seq_len is None else range(seq_len):
            pairs = product(srcs, dests)
            edge_dict[i] = [Edge(s, d, i) for s, d in pairs if s.layer < d.layer]
    nodes: Set[Node] = set(srcs | dests)
    edges = set(list(chain.from_iterable(edge_dict.values())))

    return nodes, srcs, dests, edge_dict, edges, seq_dim, seq_len


def make_model_patchable(
    model: t.nn.Module,
    factorized: bool,
    src_nodes: Set[SrcNode],
    nodes: Set[Node],
    device: t.device,
    seq_len: Optional[int] = None,
    seq_dim: Optional[int] = None,
) -> Tuple[Set[PatchWrapperImpl], Set[PatchWrapperImpl], Set[PatchWrapperImpl]]:
    """
    Injects [`PatchWrapper`][auto_circuit.types.PatchWrapper]s into the model at the
    node positions to enable patching.

    Args:
        model: The model to make patchable.
        factorized: Whether the model is factorized, for Edge Ablation. Otherwise,
            only Node Ablation is possible.
        src_nodes: The source nodes in the model.
        nodes: All the nodes in the model.
        device: The device to put the patch masks on.
        seq_len: The sequence length of the model inputs. If `None`, all token positions
            are simultaneously ablated.
        seq_dim: The sequence dimension of the model. This is the dimension on which new
            inputs are concatenated. In transformers, this is `1` because the
            activations are of shape `[batch_size, seq_len, hidden_dim]`.

    Returns:
        Tuple containing:
            <ol>
                <li>The set of all PatchWrapper modules in the model.</li>
                <li>The set of all PatchWrapper modules that wrap source nodes.</li>
                <li>The set of all PatchWrapper modules that wrap destination
                    nodes.</li>
            </ol>

    Warning:
        This function modifies the model in place.
    """
    node_dict: Dict[str, Set[Node]] = defaultdict(set)
    [node_dict[node.module_name].add(node) for node in nodes]
    wrappers, src_wrappers, dest_wrappers = set(), set(), set()
    dtype = next(model.parameters()).dtype

    for module_name, module_nodes in node_dict.items():
        module = module_by_name(model, module_name)
        src_idxs_slice = None
        a_node = next(iter(module_nodes))
        head_dim = a_node.head_dim
        assert all([node.head_dim == head_dim for node in module_nodes])

        if is_src := any([type(node) == SrcNode for node in module_nodes]):
            src_idxs = [n.src_idx for n in module_nodes if type(n) == SrcNode]
            src_idxs_slice = slice(min(src_idxs), max(src_idxs) + 1)
            assert src_idxs_slice.stop - src_idxs_slice.start == len(src_idxs)

        mask, in_srcs = None, None
        if is_dest := any([type(node) == DestNode for node in module_nodes]):
            module_dest_count = len([n for n in module_nodes if type(n) == DestNode])
            if factorized:
                n_in_src = len([n for n in src_nodes if n.layer < a_node.layer])
                n_ignore_src = 0
            else:
                n_in_src = len([n for n in src_nodes if n.layer + 1 == a_node.layer])
                n_ignore_src = len([n for n in src_nodes if n.layer + 1 < a_node.layer])
            in_srcs = slice(n_ignore_src, n_ignore_src + n_in_src)
            seq_shape = [seq_len] if seq_len is not None else []
            head_shape = [module_dest_count] if head_dim is not None else []
            mask_shape = seq_shape + head_shape + [n_in_src]
            mask = t.zeros(mask_shape, device=device, dtype=dtype, requires_grad=False)

        wrapper = PatchWrapperImpl(
            module_name=module_name,
            module=module,
            head_dim=head_dim,
            seq_dim=None if seq_len is None else seq_dim,  # Patch tokens separately
            is_src=is_src,
            src_idxs=src_idxs_slice,
            is_dest=is_dest,
            patch_mask=mask,
            in_srcs=in_srcs,
        )
        set_module_by_name(model, module_name, wrapper)
        wrappers.add(wrapper)
        src_wrappers.add(wrapper) if is_src else None
        dest_wrappers.add(wrapper) if is_dest else None

    return wrappers, src_wrappers, dest_wrappers


@contextmanager
def patch_mode(
    model: PatchableModel,
    patch_src_outs: t.Tensor,
    edges: Optional[Collection[str | Edge]] = None,
    curr_src_outs: Optional[t.Tensor] = None,
):
    """
    Context manager to enable patching in the model.

    Args:
        model: The patchable model to alter.
        patch_src_outs: The activations with which to ablate the model. Mask values
            interpolate the edge activations between the default activations (`0`) and
            these activations (`1`).
        curr_src_outs (t.Tensor, optional): Stores the outputs of each src node during
            the current forward pass. The only time this need to be initialized is when
            you are starting the forward pass at a middle layer because the outputs of
            previous
            [`SrcNode`][auto_circuit.types.SrcNode]s won't be cached automatically (used
            in ACDC, as a performance optimization).

    Warning:
        This function modifies the state of the model! This is a likely source of bugs.
    """
    if curr_src_outs is None:
        curr_src_outs = t.zeros_like(patch_src_outs)

    if edges is not None:
        set_all_masks(model, val=0.0)
        for edge in model.edges:
            if edge in edges or edge.name in edges:
                edge.patch_mask(model).data[edge.patch_idx] = 1.0

    for wrapper in model.wrappers:
        wrapper.patch_mode = True
        wrapper.curr_src_outs = curr_src_outs
        if wrapper.is_dest:
            wrapper.patch_src_outs = patch_src_outs
    try:
        yield
    finally:
        for wrapper in model.wrappers:
            wrapper.patch_mode = False
            wrapper.curr_src_outs = None
            if wrapper.is_dest:
                wrapper.patch_src_outs = None
        del curr_src_outs, patch_src_outs


def set_all_masks(model: PatchableModel, val: float) -> None:
    """
    Set all the patch masks in the model to the specified value.

    Args:
        model: The patchable model to alter.
        val: The value to set the patch masks to.

    Warning:
        This function modifies the state of the model! This is a likely source of bugs.
    """
    for wrapper in model.wrappers:
        if wrapper.is_dest:
            t.nn.init.constant_(wrapper.patch_mask, val)


@contextmanager
def train_mask_mode(
    model: PatchableModel, requires_grad: bool = True
) -> Iterator[Dict[str, t.nn.Parameter]]:
    """
    Context manager that sets the `requires_grad` attribute of the patch masks for the
    duration of the context and yields the parameters.

    Args:
        model: The patchable model to alter.
        requires_grad: Whether to enable gradient tracking on the patch masks.

    Yields:
        The patch mask `Parameter`s of the model as a dictionary with the module name as
        the key.

    Warning:
        This function modifies the state of the model! This is a likely source of bugs.
    """
    model.eval()
    model.zero_grad()
    parameters: Dict[str, t.nn.Parameter] = {}
    for wrapper in model.dest_wrappers:
        patch_mask = wrapper.patch_mask
        patch_mask.detach_().requires_grad_(requires_grad)
        parameters[wrapper.module_name] = patch_mask
        wrapper.train()
    try:
        yield parameters
    finally:
        for wrapper in model.dest_wrappers:
            wrapper.eval()
            wrapper.patch_mask.detach_().requires_grad_(False)


@contextmanager
def mask_fn_mode(model: PatchableModel, mask_fn: MaskFn, dropout_p: float = 0.0):
    """
    Context manager to enable the specified `mask_fn` and `dropout_p` for a patchable
    model.

    Args:
        model: The patchable model to alter.
        mask_fn: The function to apply to the mask values before they are used to
            interpolate between the clean and ablated activations.
        dropout_p: The dropout probability to apply to the mask values.

    Warning:
        This function modifies the state of the model! This is a likely source of bugs.
    """
    for wrapper in model.dest_wrappers:
        wrapper.mask_fn = mask_fn
        wrapper.dropout_layer.p = dropout_p  # type: ignore
    try:
        yield
    finally:
        for wrapper in model.dest_wrappers:
            wrapper.mask_fn = None
            wrapper.dropout_layer.p = 0.0  # type: ignore


def edge_counts_util(
    edges: Set[Edge],
    test_counts: Optional[TestEdges] = None,  # None means default
    prune_scores: Optional[PruneScores] = None,
    zero_edges: Optional[bool] = None,  # None means default
    all_edges: Optional[bool] = None,  # None means default
    true_edge_count: Optional[int] = None,
) -> List[int]:
    """
    Calculate a set of [number of edges in the circuit] to test.

    Args:
        edges: The set of all edges in the model (just used to count the maximum circuit
            size).
        test_counts: The method to determine the set of edge counts. If None, the
            function will try to infer the best method based on the number of edges and
            the `prune_scores`. See [`TestEdges`][auto_circuit.types.TestEdges] and
            [`EdgeCounts`][auto_circuit.types.EdgeCounts] for full details.
        prune_scores: The scores to use to determine the edge counts. Used to make a
            better inference of the best set to use when `test_counts` is None. Also
            used when `test_counts` is `EdgeCounts.GROUPS` to group the edges by their
            scores.
        zero_edges: Whether to include `0` edges.
        all_edges: Whether to include `n_edges` edges (where `n_edges` is the number of
            edges in the model).
        true_edge_count: Inserts an extra specified edge count into the list. Useful
            when you want to test the number of edges in the candidate circuit.

    Returns:
        The list of edge counts to test.
    """
    n_edges = len(edges)

    # Work out default setting for test_counts
    sorted_ps_count: Optional[t.Tensor] = None
    if test_counts is None:
        test_counts = EdgeCounts.LOGARITHMIC if n_edges > 200 else EdgeCounts.ALL
        if prune_scores is not None:
            flat_ps = desc_prune_scores(prune_scores)
            unique_ps, sorted_ps_count = flat_ps.unique(sorted=True, return_counts=True)
            if list(unique_ps.size())[0] < min(n_edges / 2, 100):
                test_counts = EdgeCounts.GROUPS

    # Calculate the test counts
    if test_counts == EdgeCounts.ALL:
        counts_list = [n for n in range(n_edges + 1)]
    elif test_counts == EdgeCounts.LOGARITHMIC:
        counts_list = [
            n
            for n in range(1, n_edges)
            # if n % (10 ** max(math.floor(math.log10(n)) - 1, 0)) == 0
            if n % (10 ** max(math.floor(math.log10(n)), 0)) == 0
        ]
    elif test_counts == EdgeCounts.GROUPS:
        assert prune_scores is not None
        if sorted_ps_count is None:
            flat_ps = desc_prune_scores(prune_scores)
            _, sorted_ps_count = flat_ps.unique(sorted=True, return_counts=True)
        assert sorted_ps_count is not None
        counts_list = sorted_ps_count.flip(dims=(0,)).cumsum(dim=0).tolist()
    elif isinstance(test_counts, List):
        counts_list = [n if type(n) == int else int(n_edges * n) for n in test_counts]
    else:
        raise NotImplementedError(f"Unknown test_counts: {test_counts}")

    # Choose default. If len(count_lists) <= 2, this is likely a binary circuit encoding
    if zero_edges is None:
        zero_edges = True if len(counts_list) > 2 else False
    if all_edges is None:
        all_edges = True if len(counts_list) > 2 else False

    # Add zero and all edges if necessary
    if zero_edges and 0 not in counts_list:
        counts_list = [0] + counts_list
    if all_edges and n_edges not in counts_list:
        counts_list.append(n_edges)
    if not zero_edges and 0 in counts_list:
        counts_list.remove(0)
    if not all_edges and n_edges in counts_list:
        counts_list.remove(n_edges)
    # Insert true_edge_count at the correct position
    if true_edge_count is not None and true_edge_count not in counts_list:
        counts_list.append(true_edge_count)
    counts_list.sort()

    return counts_list
