from functools import partial
from typing import Dict, List, Tuple

import torch as t
import transformer_lens
from ordered_set import OrderedSet

import auto_circuit.model_utils.transformer_lens_utils as tl_utils
from auto_circuit.types import Edge, EdgeDest, EdgeSrc


def graph_edges(model: t.nn.Module) -> OrderedSet[Edge]:
    """Get Edge objects for all attention heads and MLPs."""
    if isinstance(model, transformer_lens.HookedTransformer):
        src_layers: List[OrderedSet[EdgeSrc]] = tl_utils.graph_src_layers(model)
        dest_layers: List[OrderedSet[EdgeDest]] = tl_utils.graph_dest_layers(model)
    else:
        raise NotImplementedError(model)

    edges = []
    for src_layer, layer_srcs in enumerate(src_layers):
        for edge in layer_srcs:
            for layer_dests in dest_layers[src_layer:]:
                for dest in layer_dests:
                    edges.append(Edge(src=edge, dest=dest))
    return OrderedSet(edges)


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
            handles.append(edge.src.module.register_forward_hook(hook_fn))
        model(input)
    finally:
        [handle.remove() for handle in handles]
    return activations
