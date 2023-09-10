from typing import List

import transformer_lens as tl
from ordered_set import OrderedSet

from auto_circuit.types import DestNode, Edge, SrcNode


def fctrzd_graph_src_lyrs(model: tl.HookedTransformer) -> List[OrderedSet[SrcNode]]:
    """Get the source part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert model.cfg.use_split_qkv_input  # Separate Q, K, V input for each head
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers = []
    resid_start = SrcNode(
        name="Resid Start",
        module_name="blocks.0.hook_resid_pre",
        weight="embed.W_E",
    )
    layers.append(OrderedSet([resid_start]))
    for block_idx in range(model.cfg.n_layers):
        attn_set = OrderedSet([])
        for head_idx in range(model.cfg.n_heads):
            attn_src = SrcNode(
                name=f"A{block_idx}.{head_idx}",
                module_name=f"blocks.{block_idx}.attn.hook_result",
                _out_idx=(None, None, head_idx),
                weight=f"blocks.{block_idx}.attn.W_O",
                _weight_t_idx=head_idx,
            )
            attn_set.add(attn_src)
        layers.append(attn_set)
        mlp_set = OrderedSet(
            [
                SrcNode(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.mlp",
                    weight=f"blocks.{block_idx}.mlp.W_out",
                )
            ]
        )
        layers.append(mlp_set)
    return layers


def fctrzd_graph_dest_lyrs(model: tl.HookedTransformer) -> List[OrderedSet[DestNode]]:
    """Get the destination part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert model.cfg.use_split_qkv_input  # Separate Q, K, V input for each head
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers = []
    for block_idx in range(model.cfg.n_layers):
        attn_set = OrderedSet([])
        for head_idx in range(model.cfg.n_heads):
            for letter in ["Q", "K", "V"]:
                attn_dest = DestNode(
                    name=f"A{block_idx}.{head_idx}.{letter}",
                    module_name=f"blocks.{block_idx}.hook_{letter.lower()}_input",
                    _in_idx=(None, None, head_idx),
                    weight=f"blocks.{block_idx}.attn.W_{letter}",
                    _weight_t_idx=head_idx,
                )
                attn_set.add(attn_dest)
        layers.append(attn_set)
        mlp_dest = DestNode(
            name=f"MLP {block_idx}",
            module_name=f"blocks.{block_idx}.hook_mlp_in",
            weight=f"blocks.{block_idx}.mlp.W_in",
        )
        layers.append(set([mlp_dest]))
    resid_end = DestNode(
        name="Resid End",
        module_name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
        weight="unembed.W_U",
    )
    layers.append(set([resid_end]))
    return layers


def simple_graph_edges(model: tl.HookedTransformer) -> OrderedSet[Edge]:
    """Get the edges in the unfactorized graph."""
    edges: List[Edge] = []
    for block_idx in range(model.cfg.n_layers):
        attn_srcs = []
        for head_idx in range(model.cfg.n_heads):
            attn_src = SrcNode(
                name=f"A{block_idx}.{head_idx}",
                module_name=f"blocks.{block_idx}.attn.hook_result",
                _out_idx=(None, None, head_idx),
                weight=f"blocks.{block_idx}.attn.W_O",
                _weight_t_idx=head_idx,
            )
            attn_srcs.append(attn_src)
        resid_mid_dest = DestNode(
            name=f"Block {block_idx} Resid Mid",
            module_name=f"blocks.{block_idx}.hook_resid_mid",
        )
        for attn_src in attn_srcs:
            edges.append(Edge(src=attn_src, dest=resid_mid_dest))

        mlp_src = SrcNode(
            name=f"MLP {block_idx}",
            module_name=f"blocks.{block_idx}.mlp",
            weight=f"blocks.{block_idx}.mlp.W_out",
        )
        last_block = block_idx + 1 == model.cfg.n_layers
        resid_post_dest = DestNode(
            name="Resid Final" if last_block else f"Block {block_idx} Resid Post",
            module_name=f"blocks.{block_idx}.hook_resid_post",
        )
        edges.append(Edge(src=mlp_src, dest=resid_post_dest))
    return OrderedSet(edges)
