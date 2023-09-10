import math
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import pygraphviz as pgv
import torch as t
import transformer_lens
from ordered_set import OrderedSet

import auto_circuit.model_utils.micro_model_utils as mm_utils
import auto_circuit.model_utils.transformer_lens_utils as tl_utils
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.types import DestNode, Edge, EdgeCounts, SrcNode, TestEdges
from auto_circuit.utils.misc import remove_hooks


def graph_edges(model: t.nn.Module, factorized: bool) -> OrderedSet[Edge]:
    """Get the edges of the computation graph of the model."""
    if not factorized:
        if isinstance(model, MicroModel):
            return mm_utils.simple_graph_edges(model)
        elif isinstance(model, transformer_lens.HookedTransformer):
            return tl_utils.simple_graph_edges(model)
        else:
            raise NotImplementedError(model)
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
        for src_layer, layer_srcs in enumerate(src_lyrs):
            for edge in layer_srcs:
                for layer_dests in dest_lyrs[src_layer:]:
                    for dest in layer_dests:
                        edges.append(Edge(src=edge, dest=dest))
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


def get_act_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    node: SrcNode,
    act_dict: Dict[SrcNode, t.Tensor],
) -> None:
    act_dict[node] = output[node.out_idx]


def get_src_outs(
    model: t.nn.Module, nodes: Set[SrcNode], input: t.Tensor
) -> Dict[SrcNode, t.Tensor]:
    node_outs: Dict[SrcNode, t.Tensor] = {}
    with remove_hooks() as handles:
        for node in nodes:
            hook_fn = partial(get_act_hook, node=node, act_dict=node_outs)
            handles.append(node.module(model).register_forward_hook(hook_fn))
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
            handles.append(node.module(model).register_forward_pre_hook(hook_fn))
        with t.inference_mode():
            model(input)
    return node_inputs


def parse_name(n: str) -> str:
    return n[:-2] if n.startswith("A") and len(n) > 4 else n
    # return n.split(".")[0] if n.startswith("A") else n


def draw_graph(
    model: t.nn.Module,
    factorized: bool,
    input: t.Tensor,
    patched_edges: Optional[List[Edge]] = None,
    patch_src_outs: Optional[Dict[SrcNode, t.Tensor]] = None,
) -> None:
    edges = graph_edges(model, factorized)
    src_nodes = graph_src_nodes(model, factorized)
    dest_nodes = graph_dest_nodes(model, factorized)
    G = pgv.AGraph(strict=False, directed=True)

    src_outs: Dict[SrcNode, t.Tensor] = get_src_outs(model, src_nodes, input)
    dest_ins: Dict[DestNode, t.Tensor] = get_dest_ins(model, dest_nodes, input)
    node_ins: Dict[str, t.Tensor] = dict([(k.name, v) for k, v in dest_ins.items()])
    node_ins = defaultdict(lambda: t.tensor([0.0]), node_ins)

    for edge in edges:
        patched_edge = False
        if patched_edges is not None and edge in patched_edges:
            assert patch_src_outs is not None and edge.src in patch_src_outs
            patched_edge = True
            edge_act = patch_src_outs[edge.src]
        else:
            edge_act = src_outs[edge.src]
        G.add_edge(
            edge.src.name + f"\n{str(node_ins[edge.src.name].tolist())}",
            edge.dest.name + f"\n{str(node_ins[edge.dest.name].tolist())}",
            label=str(edge_act.tolist()),
            color="red" if patched_edge else "black",
        )

    G.layout(prog="dot")  # neato
    G.draw("graphviz.png")
