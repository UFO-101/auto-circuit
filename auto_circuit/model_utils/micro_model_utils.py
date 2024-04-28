"""
This module defines a tiny toy model used mostly for testing purposes.
"""
from itertools import count
from typing import Set, Tuple

import einops
import torch as t

from auto_circuit.types import DestNode, SrcNode


class Block(t.nn.Module):
    """Trivial linear layer with input and output of size 2."""

    def __init__(self):
        super().__init__()
        self.head_inputs = t.nn.Identity()
        self.head_outputs = t.nn.Identity()
        self.weights = t.nn.Parameter(t.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def forward(self, x: t.Tensor) -> t.Tensor:  # shape: (batch, resid)
        x = einops.repeat(x, "b s r -> b s h r", h=2)
        x = self.head_inputs(x)
        x = einops.einsum(self.weights, x, "h r, b s h r -> b s h r")
        x = self.head_outputs(x)
        return einops.einsum(x, "b s h r -> b s r")


class MicroModel(t.nn.Module):
    """A trivial model with two "heads" per layer that perform simple multiplication."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.input = t.nn.Identity()
        self.n_layers = n_layers
        self.blocks, self.resids = t.nn.ModuleList(), t.nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(Block())
            self.resids.append(t.nn.Identity())
        self.output = t.nn.Identity()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.input(x)  # shape: (batch, seq, resid)
        for layer_idx in range(self.n_layers):
            block_out = self.blocks[layer_idx](x)  # shape: (batch, seq, resid)
            x = self.resids[layer_idx](x + block_out)
        return self.output(x)  # shape: (batch, seq, resid)


def factorized_src_nodes(model: MicroModel) -> Set[SrcNode]:
    """
    Get the source part of each edge in the factorized graph, grouped by layer.

    Used by [`graph_edges`][auto_circuit.utils.graph_utils.graph_edges] in
    [`patchable_model`][auto_circuit.utils.graph_utils.patchable_model].
    """
    nodes = set()
    layers, idxs = count(), count()
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="input",
            layer=next(layers),
            src_idx=next(idxs),
        )
    )
    for layer_idx in range(model.n_layers):
        layer = next(layers)
        for elem in [0, 1]:
            nodes.add(
                SrcNode(
                    name=f"B{layer_idx}.{elem}",
                    module_name=f"blocks.{layer_idx}.head_outputs",
                    layer=layer,
                    src_idx=next(idxs),
                    head_idx=elem,
                    head_dim=2,
                    weight="weights",
                    weight_head_dim=0,
                )
            )
    return nodes


def factorized_dest_nodes(model: MicroModel) -> Set[DestNode]:
    """
    Get the destination part of each edge in the factorized graph, grouped by layer.

    Used by [`graph_edges`][auto_circuit.utils.graph_utils.graph_edges] in
    [`patchable_model`][auto_circuit.utils.graph_utils.patchable_model].
    """
    nodes = set()
    layers = count(1)
    for layer_idx in range(model.n_layers):
        layer = next(layers)
        for elem in [0, 1]:
            nodes.add(
                DestNode(
                    name=f"B{layer_idx}.{elem}",
                    module_name=f"blocks.{layer_idx}.head_inputs",
                    layer=layer,
                    head_idx=elem,
                    head_dim=2,
                )
            )
    nodes.add(
        DestNode(
            name="Resid End",
            module_name="output",
            layer=next(layers),
        )
    )
    return nodes


def simple_graph_nodes(model: MicroModel) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """
    Get the nodes of the unfactorized graph.

    [`graph_edges`][auto_circuit.utils.graph_utils.graph_edges] requires that all input
    [`SrcNodes`][auto_circuit.types.SrcNode] are in the previous layer to the respective
    [`DestNodes`][auto_circuit.types.DestNode].
    """
    src_nodes, dest_nodes = set(), set()
    layers, src_idxs = count(), count()
    layer = next(layers)
    for layer_idx in range(model.n_layers):
        min_src_idx = next(src_idxs)
        first_block = layer_idx == 0
        src_nodes.add(
            SrcNode(
                name="Resid Start" if first_block else f"Resid Post {layer_idx -1}",
                module_name="input" if first_block else f"resids.{layer_idx - 1}",
                layer=layer,
                src_idx=min_src_idx,
            )
        )
        for elem in [0, 1]:
            src_nodes.add(
                SrcNode(
                    name=f"B{layer_idx}.{elem}",
                    module_name=f"blocks.{layer_idx}.head_outputs",
                    layer=layer,
                    src_idx=next(src_idxs),
                    head_idx=elem,
                    head_dim=2,
                    weight="weights",
                    weight_head_dim=0,
                )
            )
        last_block = layer_idx == model.n_layers - 1
        layer = next(layers)
        dest_nodes.add(
            DestNode(
                name="Resid End" if last_block else f"Resid Post {layer_idx}",
                module_name="output" if last_block else f"resids.{layer_idx}",
                layer=layer,
                min_src_idx=min_src_idx,
            )
        )
    return src_nodes, dest_nodes
