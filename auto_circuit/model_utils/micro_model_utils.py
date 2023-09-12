from typing import List

import einops
import torch as t
from ordered_set import OrderedSet

from auto_circuit.types import DestNode, Edge, SrcNode


class Block(t.nn.Module):
    """Trivial linear layer with input and output of size 2."""

    def __init__(self, weights: t.Tensor):
        super(Block, self).__init__()
        self.weights = weights  # shape: (2, 2)
        self.head_inputs = t.nn.Identity()
        self.head_outputs = t.nn.Identity()

    def forward(self, x: t.Tensor) -> t.Tensor:  # shape: (batch, resid)
        x = einops.repeat(x, "b r -> b h r", h=2)
        x = self.head_inputs(x)
        x = einops.einsum(self.weights, x, "h r, b h r -> b h r")
        x = self.head_outputs(x)
        return einops.einsum(x, "b h r -> b r")


class MicroModel(t.nn.Module):
    """A model trivial with three layers of simple operations."""

    def __init__(self, n_layers: int = 2):
        super(MicroModel, self).__init__()
        self.input = t.nn.Identity()
        self.n_layers = n_layers
        self.blocks, self.resids = t.nn.ModuleList(), t.nn.ModuleList()
        self.weights = t.tensor([[1.0, 2.0], [3.0, 4.0]])
        for _ in range(n_layers):
            self.blocks.append(Block(weights=self.weights))
            self.resids.append(t.nn.Identity())
        self.output = t.nn.Identity()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.input(x)
        for layer_idx in range(self.n_layers):
            block_out = self.blocks[layer_idx](x)
            x = self.resids[layer_idx](x + block_out)
        return self.output(x)


def fctrzd_graph_src_lyrs(model: MicroModel) -> List[OrderedSet[SrcNode]]:
    """Get the source part of each edge in the factorized graph, grouped by layer."""
    layers = []
    layers.append(OrderedSet([SrcNode(name="Input", module_name="input")]))
    for layer_idx in range(model.n_layers):
        mul_set = OrderedSet([])
        for elem in [0, 1]:
            mul_set.add(
                SrcNode(
                    name=f"Block Layer {layer_idx} Elem {elem}",
                    module_name=f"blocks.{layer_idx}.head_outputs",
                    _out_idx=(None, elem),
                    weight="weights",
                    _weight_t_idx=elem,
                )
            )
        layers.append(mul_set)
    return layers


def fctrzd_graph_dest_lyrs(model: MicroModel) -> List[OrderedSet[DestNode]]:
    layers = []
    for layer_idx in range(model.n_layers):
        elem_set = OrderedSet([])
        for elem in [0, 1]:
            elem_set.add(
                DestNode(
                    name=f"Block Layer {layer_idx} Elem {elem}",
                    module_name=f"blocks.{layer_idx}.head_inputs",
                    _in_idx=(None, elem),
                )
            )
        layers.append(elem_set)
    layers.append(OrderedSet([DestNode(name="Output", module_name="output")]))
    return layers


def simple_graph_edges(model: MicroModel) -> OrderedSet[Edge]:
    edges = []
    for layer_idx in range(model.n_layers):
        elem_set = []
        for elem in [0, 1]:
            elem_set.append(
                SrcNode(
                    name=f"Block Layer {layer_idx} Elem {elem}",
                    module_name=f"blocks.{layer_idx}.head_inputs",
                    _out_idx=(None, elem),
                    weight="weights",
                    _weight_t_idx=elem,
                )
            )
        last_block = layer_idx == model.n_layers - 1
        resid = DestNode(
            name="Output" if last_block else f"Resid Layer {layer_idx}",
            module_name="output" if last_block else f"resids.{layer_idx}",
        )
        for elem in elem_set:
            edges.append(Edge(src=elem, dest=resid))
    return OrderedSet(edges)
