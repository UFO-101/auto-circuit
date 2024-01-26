#%%
import math
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import accumulate, chain, product
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
    DestNode,
    Edge,
    EdgeCounts,
    Node,
    SrcNode,
    TestEdges,
)
from auto_circuit.utils.misc import module_by_name, remove_hooks, set_module_by_name
from auto_circuit.utils.patch_wrapper import PatchWrapper
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import MaskFn

#%%


def patchable_model(
    model: t.nn.Module,
    factorized: bool,
    separate_qkv: bool,
    slice_output: bool = False,
    seq_len: Optional[int] = None,
    kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    device: t.device = t.device("cpu"),
) -> PatchableModel:
    nodes, srcs, dests, edge_dict, edges, seq_dim, seq_len = graph_edges(
        model, factorized, separate_qkv, seq_len
    )
    wrappers, src_wrappers, dest_wrappers = make_model_patchable(
        model, srcs, nodes, device, seq_len, seq_dim
    )
    out_slice: Tuple[slice | int, ...] = (
        tuple([slice(None)] * seq_dim + [-1]) if slice_output else (slice(None),)
    )
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
        kv_cache=kv_cache,
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
) -> Tuple[Set[PatchWrapper], Set[PatchWrapper], Set[PatchWrapper]]:
    node_dict: Dict[str, Set[Node]] = defaultdict(set)
    [node_dict[node.module_name].add(node) for node in nodes]
    wrappers, src_wrappers, dest_wrappers = set(), set(), set()

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

        patch_mask, prev_src_count = None, None
        if is_dest := any([type(node) == DestNode for node in module_nodes]):
            module_dest_count = len([n for n in module_nodes if type(n) == DestNode])
            prev_src_count = len([n for n in src_nodes if n.layer < a_node.layer])
            seq_shape = [seq_len] if seq_len is not None else []
            head_shape = [module_dest_count] if module_dest_count > 1 else []
            mask_shape = seq_shape + head_shape + [prev_src_count]
            patch_mask = t.zeros(mask_shape, device=device)
        wrapper = PatchWrapper(
            module=module,
            head_dim=head_dim,
            seq_dim=None if seq_len is None else seq_dim,  # Patch tokens separately
            is_src=is_src,
            src_idxs=src_idxs,
            is_dest=is_dest,
            patch_mask=patch_mask,
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
def train_mask_mode(model: PatchableModel) -> Iterator[List[t.nn.Parameter]]:
    model.eval()
    model.zero_grad()
    parameters: List[t.nn.Parameter] = []
    for wrapper in model.dest_wrappers:
        parameters.append(wrapper.patch_mask)
        wrapper.train()
    try:
        yield parameters
    finally:
        for wrapper in model.dest_wrappers:
            wrapper.eval()


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
    prune_scores: Optional[Dict[Edge, float]] = None,
    zero_edges: Optional[bool] = None,  # None means default
    all_edges: Optional[bool] = None,  # None means default
) -> List[int]:
    n_edges = len(edges)

    # Work out default setting for test_counts
    if test_counts is None:
        if prune_scores is not None and len(set(prune_scores.values())) < n_edges / 2:
            test_counts = EdgeCounts.GROUPS
        else:
            test_counts = EdgeCounts.LOGARITHMIC if n_edges > 200 else EdgeCounts.ALL

    # Calculate the test counts
    if test_counts == EdgeCounts.ALL:
        counts_list = [n for n in range(n_edges + 1)]
    elif test_counts == EdgeCounts.LOGARITHMIC:
        counts_list = [
            n
            for n in range(1, n_edges)
            if n % (10 ** max(math.floor(math.log10(n)) - 1, 0)) == 0
        ]
    elif test_counts == EdgeCounts.GROUPS:
        assert prune_scores is not None
        score_counts = Counter(prune_scores.values())
        sorted_counts = sorted(score_counts.items(), key=lambda x: x[0], reverse=True)
        counts_list = list(accumulate([n for _, n in sorted_counts]))
    elif isinstance(test_counts, List):
        counts_list = [n if type(n) == int else int(n_edges * n) for n in test_counts]
    else:
        raise NotImplementedError(f"Unknown test_counts: {test_counts}")

    # Work out default setting for zero_edges and all_edges
    zero_edges = True if len(counts_list) > 1 and zero_edges is None else zero_edges
    all_edges = True if len(counts_list) > 1 and all_edges is None else all_edges

    # Add zero and all edges if necessary
    if zero_edges and 0 not in counts_list:
        counts_list = [0] + counts_list
    if all_edges and n_edges not in counts_list:
        counts_list.append(n_edges)

    return counts_list


def src_out_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    out: t.Tensor,
    src: SrcNode,
    src_outs: Dict[SrcNode, t.Tensor],
):
    out = out if src.head_dim is None else out.split(1, dim=src.head_dim)[src.head_idx]
    src_outs[src] = out if src.head_dim is None else out.squeeze(src.head_dim)


# TODO optimize this to use one hook per src module
def get_sorted_src_outs(
    model: PatchableModel,
    input: t.Tensor,
) -> Dict[SrcNode, t.Tensor]:
    src_outs: Dict[SrcNode, t.Tensor] = {}
    with remove_hooks() as handles:
        for node in model.srcs:
            hook_fn = partial(src_out_hook, src=node, src_outs=src_outs)
            handles.add(node.module(model).register_forward_hook(hook_fn))
        with t.inference_mode():
            model(input)
    src_outs = dict(sorted(src_outs.items(), key=lambda x: x[0].idx))
    assert [src.idx for src in src_outs.keys()] == list(range(len(src_outs)))
    return src_outs


def dest_in_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    dest: DestNode,
    dest_ins: Dict[DestNode, t.Tensor],
):
    in_0 = input[0] if type(input) == tuple else input
    assert type(in_0) == t.Tensor
    in_0 = (
        in_0
        if dest.head_dim is None
        else in_0.split(1, dim=dest.head_dim)[dest.head_idx]
    )
    dest_ins[dest] = in_0 if dest.head_dim is None else in_0.squeeze(dest.head_dim)


def get_sorted_dest_ins(
    model: PatchableModel,
    input: t.Tensor,
) -> Dict[DestNode, t.Tensor]:
    dest_ins: Dict[DestNode, t.Tensor] = {}
    with remove_hooks() as handles:
        for node in model.dests:
            hook_fn = partial(dest_in_hook, dest=node, dest_ins=dest_ins)
            handles.add(node.module(model).module.register_forward_hook(hook_fn))
        with t.inference_mode():
            model(input)
    dest_ins = dict(sorted(dest_ins.items(), key=lambda x: x[0].idx))
    assert [dest.idx for dest in dest_ins.keys()] == list(range(len(dest_ins)))
    return dest_ins
