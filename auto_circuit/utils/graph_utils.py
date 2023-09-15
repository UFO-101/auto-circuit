import math
from functools import partial
from typing import Dict, List, Set, Tuple

import torch as t
import transformer_lens
from ordered_set import OrderedSet

import auto_circuit.model_utils.micro_model_utils as mm_utils
import auto_circuit.model_utils.transformer_lens_utils as tl_utils
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.types import DestNode, Edge, EdgeCounts, SrcNode, TestEdges
from auto_circuit.utils.misc import remove_hooks


def graph_edges(
    model: t.nn.Module, factorized: bool, reverse_topo_sort: bool = False
) -> OrderedSet[Edge]:
    """Get the edges of the computation graph of the model."""
    if not factorized:
        if isinstance(model, MicroModel):
            edge_set: OrderedSet[Edge] = mm_utils.simple_graph_edges(model)
        elif isinstance(model, transformer_lens.HookedTransformer):
            edge_set: OrderedSet[Edge] = tl_utils.simple_graph_edges(model)
        else:
            raise NotImplementedError(model)

        if reverse_topo_sort:
            edge_set = OrderedSet([edge for edge in edge_set][::-1])
        return edge_set
    else:
        if isinstance(model, MicroModel):
            src_lyrs: List[OrderedSet[SrcNode]] = mm_utils.fctrzd_graph_src_lyrs(model)
            dest_lyrs: List[OrderedSet[DestNode]] = mm_utils.fctrzd_graph_dest_lyrs(
                model
            )
        elif isinstance(model, transformer_lens.HookedTransformer):
            src_lyrs: List[OrderedSet[SrcNode]] = tl_utils.fctrzd_graph_src_lyrs(model)
            dest_lyrs: List[OrderedSet[DestNode]] = tl_utils.fctrzd_graph_dest_lyrs(
                model
            )
        else:
            raise NotImplementedError(model)

        edges = []
        if reverse_topo_sort is False:
            for src_layer, layer_srcs in enumerate(src_lyrs):
                for src in layer_srcs:
                    for layer_dests in dest_lyrs[src_layer:]:
                        for dest in layer_dests:
                            edges.append(Edge(src=src, dest=dest))
        else:
            for dest_layer, layer_dests in list(enumerate(dest_lyrs))[::-1]:
                for dest in layer_dests[::-1]:
                    for layer_srcs in src_lyrs[dest_layer::-1]:
                        for src in layer_srcs[::-1]:
                            edges.append(Edge(dest=dest, src=src))
        return OrderedSet(edges)


def graph_src_nodes(model: t.nn.Module, factorized: bool) -> Set[SrcNode]:
    """Get the src nodes of the computational graph of the model."""
    edges = graph_edges(model, factorized)
    return set([edge.src for edge in edges])


def graph_dest_nodes(model: t.nn.Module, factorized: bool) -> Set[DestNode]:
    """Get the dest nodes of the computational graph of the model."""
    edges = graph_edges(model, factorized)
    return set([edge.dest for edge in edges])


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


def src_out_hook(
    model: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    edge_src: SrcNode,
    src_outs: Dict[SrcNode, t.Tensor],
):
    src_outs[edge_src] = output[edge_src.out_idx]


def get_src_outs(
    model: t.nn.Module, nodes: Set[SrcNode], input: t.Tensor
) -> Dict[SrcNode, t.Tensor]:
    node_outs: Dict[SrcNode, t.Tensor] = {}
    with remove_hooks() as handles:
        for node in nodes:
            hook_fn = partial(src_out_hook, edge_src=node, src_outs=node_outs)
            handles.add(node.module(model).register_forward_hook(hook_fn))
        with t.inference_mode():
            model(input)
    return node_outs


def input_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    node: DestNode,
    input_dict: Dict[DestNode, t.Tensor],
) -> None:
    input_dict[node] = input[0][node.in_idx]


def get_dest_ins(
    model: t.nn.Module, nodes: Set[DestNode], input: t.Tensor
) -> Dict[DestNode, t.Tensor]:
    node_inputs: Dict[DestNode, t.Tensor] = {}
    with remove_hooks() as handles:
        for node in nodes:
            hook_fn = partial(input_hook, node=node, input_dict=node_inputs)
            handles.add(node.module(model).register_forward_pre_hook(hook_fn))
        with t.inference_mode():
            model(input)
    return node_inputs
