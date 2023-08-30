import math
from functools import partial
from typing import Dict, List, Tuple

import torch as t
import transformer_lens
from ordered_set import OrderedSet

import auto_circuit.model_utils.transformer_lens_utils as tl_utils
from auto_circuit.types import Edge, EdgeCounts, EdgeDest, EdgeSrc, TestEdges


def graph_edges(model: t.nn.Module, factorized: bool) -> OrderedSet[Edge]:
    """Get Edge objects for all attention heads and MLPs."""
    if not factorized:
        if isinstance(model, transformer_lens.HookedTransformer):
            return tl_utils.simple_graph_edges(model)
        else:
            raise NotImplementedError(model)
    else:
        if isinstance(model, transformer_lens.HookedTransformer):
            src_lyrs: List[OrderedSet[EdgeSrc]] = tl_utils.fctrzd_graph_src_lyrs(model)
            dest_lyrs: List[OrderedSet[EdgeDest]] = tl_utils.fctrzd_graph_dest_lyrs(
                model
            )
        else:
            raise NotImplementedError(model)

        edges = []
        for src_layer, layer_srcs in enumerate(src_lyrs):
            for edge in layer_srcs:
                for layer_dests in dest_lyrs[src_layer:]:
                    for dest in layer_dests:
                        edges.append(Edge(src=edge, dest=dest))
        return OrderedSet(edges)


def edge_counts_util(
    model: t.nn.Module,
    factorized: bool,
    test_counts: TestEdges,
) -> List[int]:
    edges = graph_edges(model, factorized)
    n_edges = len(edges)

    if test_counts == EdgeCounts.ALL:
        counts_list = [n for n in range(n_edges + 1)]
    elif test_counts == EdgeCounts.LOGARITHMIC:
        counts_list = [
            n
            for n in range(n_edges + 1)
            if n % 10 ** math.floor(math.log10(max(n, 1))) == 0
        ]
    elif isinstance(test_counts, List):
        counts_list = [n if type(n) == int else int(n_edges * n) for n in test_counts]
    else:
        raise NotImplementedError(f"Unknown test_counts: {test_counts}")

    return counts_list


def output_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    edge: Edge,
    act_dict: Dict[EdgeSrc, t.Tensor],
) -> None:
    act_dict[edge.src] = output[edge.src.t_idx]


def edge_acts(
    model: t.nn.Module, edges: OrderedSet[Edge], input: t.Tensor
) -> Dict[EdgeSrc, t.Tensor]:
    activations: Dict[EdgeSrc, t.Tensor] = {}
    handles = []
    try:
        for edge in edges:
            hook_fn = partial(output_hook, edge=edge, act_dict=activations)
            handles.append(edge.src.module(model).register_forward_hook(hook_fn))
        with t.inference_mode():
            model(input)
    finally:
        [handle.remove() for handle in handles]
    return activations
