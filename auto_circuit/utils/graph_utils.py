#%%
import math
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import chain, product
from typing import Dict, Iterator, List, Optional, Set, Tuple

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
    AblationType,
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
from auto_circuit.utils.misc import module_by_name, remove_hooks, set_module_by_name
from auto_circuit.utils.patch_wrapper import PatchWrapperImpl
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import flat_prune_scores


def patchable_model(
    model: t.nn.Module,
    factorized: bool,
    separate_qkv: bool,
    slice_output: OutputSlice = None,
    seq_len: Optional[int] = None,
    kv_caches: Tuple[Optional[HookedTransformerKeyValueCache], ...] = (None,),
    device: t.device = t.device("cpu"),
) -> PatchableModel:
    nodes, srcs, dests, edge_dict, edges, seq_dim, seq_len = graph_edges(
        model, factorized, separate_qkv, seq_len
    )
    wrappers, src_wrappers, dest_wrappers = make_model_patchable(
        model, srcs, nodes, device, seq_len, seq_dim
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
        is_transformer=is_transformer,
        kv_caches=kv_caches,
        wrapped_model=model,
    )


def graph_edges(
    model: t.nn.Module,
    factorized: bool,
    separate_qkv: bool,
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
    """Get the edges of the computation graph of the model."""
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
            srcs: Set[SrcNode] = tl_utils.factorized_src_nodes(model)
            dests: Set[DestNode] = tl_utils.factorized_dest_nodes(model, separate_qkv)
        elif isinstance(model, AutoencoderTransformer):
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
    src_nodes: Set[SrcNode],
    nodes: Set[Node],
    device: t.device,
    seq_len: Optional[int] = None,
    seq_dim: Optional[int] = None,
) -> Tuple[Set[PatchWrapperImpl], Set[PatchWrapperImpl], Set[PatchWrapperImpl]]:
    node_dict: Dict[str, Set[Node]] = defaultdict(set)
    [node_dict[node.module_name].add(node) for node in nodes]
    wrappers, src_wrappers, dest_wrappers = set(), set(), set()
    dtype = next(model.parameters()).dtype

    for module_name, module_nodes in node_dict.items():
        module = module_by_name(model, module_name)
        src_idxs = None
        a_node = next(iter(module_nodes))
        head_dim = a_node.head_dim
        assert all([node.head_dim == head_dim for node in module_nodes])

        if is_src := any([type(node) == SrcNode for node in module_nodes]):
            if len(module_nodes) == 1:
                src_idxs = slice(a_node.idx, a_node.idx + 1)
            else:
                idxs = [node.idx for node in module_nodes]
                src_idxs = slice(min(idxs), max(idxs) + 1)
                assert src_idxs.stop - src_idxs.start == len(idxs)

        mask, prev_src_count = None, None
        if is_dest := any([type(node) == DestNode for node in module_nodes]):
            module_dest_count = len([n for n in module_nodes if type(n) == DestNode])
            prev_src_count = len([n for n in src_nodes if n.layer < a_node.layer])
            seq_shape = [seq_len] if seq_len is not None else []
            head_shape = [module_dest_count] if head_dim is not None else []
            mask_shape = seq_shape + head_shape + [prev_src_count]
            mask = t.zeros(mask_shape, device=device, dtype=dtype, requires_grad=False)
        wrapper = PatchWrapperImpl(
            module_name=module_name,
            module=module,
            head_dim=head_dim,
            seq_dim=None if seq_len is None else seq_dim,  # Patch tokens separately
            is_src=is_src,
            src_idxs=src_idxs,
            is_dest=is_dest,
            patch_mask=mask,
            prev_src_count=prev_src_count,
        )
        set_module_by_name(model, module_name, wrapper)
        wrappers.add(wrapper)
        src_wrappers.add(wrapper) if is_src else None
        dest_wrappers.add(wrapper) if is_dest else None

    return wrappers, src_wrappers, dest_wrappers


@contextmanager
def patch_mode(
    model: PatchableModel,
    curr_src_outs: t.Tensor,
    patch_src_outs: t.Tensor,
):
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
    for wrapper in model.wrappers:
        if wrapper.is_dest:
            t.nn.init.constant_(wrapper.patch_mask, val)


@contextmanager
def train_mask_mode(
    model: PatchableModel, requires_grad: bool = True
) -> Iterator[Dict[str, t.nn.Parameter]]:
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
    zero_edges: bool = True,
    all_edges: bool = True,
) -> List[int]:
    n_edges = len(edges)

    # Work out default setting for test_counts
    sorted_ps_count: Optional[t.Tensor] = None
    if test_counts is None:
        test_counts = EdgeCounts.LOGARITHMIC if n_edges > 200 else EdgeCounts.ALL
        if prune_scores is not None:
            flat_ps = flat_prune_scores(prune_scores)
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
            flat_ps = flat_prune_scores(prune_scores)
            _, sorted_ps_count = flat_ps.unique(sorted=True, return_counts=True)
        assert sorted_ps_count is not None
        counts_list = sorted_ps_count.flip(dims=(0,)).cumsum(dim=0).tolist()
    elif isinstance(test_counts, List):
        counts_list = [n if type(n) == int else int(n_edges * n) for n in test_counts]
    else:
        raise NotImplementedError(f"Unknown test_counts: {test_counts}")

    # Add zero and all edges if necessary
    if zero_edges and 0 not in counts_list:
        counts_list = [0] + counts_list
    if all_edges and n_edges not in counts_list:
        counts_list.append(n_edges)
    if not zero_edges and 0 in counts_list:
        counts_list.remove(0)
    if not all_edges and n_edges in counts_list:
        counts_list.remove(n_edges)

    return counts_list


def src_out_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    out: t.Tensor,
    src_nodes: List[SrcNode],
    src_outs: Dict[SrcNode, t.Tensor],
    ablation_type: AblationType,
):
    if ablation_type == AblationType.RESAMPLE:
        out = out
    elif ablation_type == AblationType.ZERO:
        out = t.zeros_like(out)
    elif ablation_type == AblationType.MEAN:
        repeats = [out.size(0)] + [1] * (out.ndim - 1)
        out = out.mean(dim=0, keepdim=True).repeat(repeats)
    else:
        raise NotImplementedError(ablation_type)

    head_dim: Optional[int] = src_nodes[0].head_dim
    out = out if head_dim is None else out.split(1, dim=head_dim)
    for s in src_nodes:
        src_outs[s] = out if head_dim is None else out[s.head_idx].squeeze(s.head_dim)


def get_sorted_src_outs(
    model: PatchableModel,
    input: t.Tensor,
    ablation_type: AblationType = AblationType.RESAMPLE,
) -> Dict[SrcNode, t.Tensor]:
    src_outs: Dict[SrcNode, t.Tensor] = {}
    src_modules: Dict[t.nn.Module, List[SrcNode]] = defaultdict(list)
    [src_modules[src.module(model)].append(src) for src in model.srcs]
    with remove_hooks() as handles:
        for mod, src_nodes in src_modules.items():
            hook_fn = partial(
                src_out_hook,
                src_nodes=src_nodes,
                src_outs=src_outs,
                ablation_type=ablation_type,
            )
            handles.add(mod.register_forward_hook(hook_fn))
        with t.inference_mode():
            model(input)
    src_outs = dict(sorted(src_outs.items(), key=lambda x: x[0].idx))
    assert [src.idx for src in src_outs.keys()] == list(range(len(src_outs)))
    return src_outs
