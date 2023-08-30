from typing import List

import transformer_lens as tl
from ordered_set import OrderedSet

from auto_circuit.types import Edge, EdgeDest, EdgeSrc


def fctrzd_graph_src_lyrs(model: tl.HookedTransformer) -> List[OrderedSet[EdgeSrc]]:
    """Get the source part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert model.cfg.use_split_qkv_input  # Get Q, K, V for each head separately
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers = []
    resid_start = EdgeSrc(
        name="Resid Start",
        module_name="blocks.0.hook_resid_pre",
        _t_idx=None,
        weight="embed.W_E",
        _weight_t_idx=None,
    )
    layers.append(OrderedSet([resid_start]))
    for block_idx in range(model.cfg.n_layers):
        attn_set = OrderedSet([])
        for head_idx in range(model.cfg.n_heads):
            attn_src = EdgeSrc(
                name=f"A{block_idx}.{head_idx}",
                module_name=f"blocks.{block_idx}.attn.hook_result",
                _t_idx=(None, None, head_idx),
                weight=f"blocks.{block_idx}.attn.W_O",
                _weight_t_idx=head_idx,
            )
            attn_set.add(attn_src)
        layers.append(attn_set)
        mlp_set = OrderedSet(
            [
                EdgeSrc(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.mlp",
                    _t_idx=None,
                    weight=f"blocks.{block_idx}.mlp.W_out",
                    _weight_t_idx=None,
                )
            ]
        )
        layers.append(mlp_set)
    return layers


def fctrzd_graph_dest_lyrs(model: tl.HookedTransformer) -> List[OrderedSet[EdgeDest]]:
    """Get the destination part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert model.cfg.use_split_qkv_input  # Get Q, K, V for each head separately
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers = []
    for block_idx in range(model.cfg.n_layers):
        attn_set = OrderedSet([])
        for head_idx in range(model.cfg.n_heads):
            for letter in ["Q", "K", "V"]:
                attn_dest = EdgeDest(
                    name=f"A{block_idx}.{head_idx}.{letter}",
                    module_name=f"blocks.{block_idx}.hook_{letter.lower()}_input",
                    _t_idx=(None, None, head_idx),
                    weight=f"blocks.{block_idx}.attn.W_{letter}",
                    _weight_t_idx=head_idx,
                )
                attn_set.add(attn_dest)
        layers.append(attn_set)
        mlp_dest = EdgeDest(
            name=f"MLP {block_idx}",
            module_name=f"blocks.{block_idx}.hook_mlp_in",
            _t_idx=None,
            weight=f"blocks.{block_idx}.mlp.W_in",
            _weight_t_idx=None,
        )
        layers.append(set([mlp_dest]))
    resid_end = EdgeDest(
        name="Resid End",
        module_name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
        _t_idx=None,
        weight="unembed.W_U",
        _weight_t_idx=None,
    )
    layers.append(set([resid_end]))
    return layers


def simple_graph_edges(model: tl.HookedTransformer) -> OrderedSet[Edge]:
    """Get the edges in the unfactorized graph."""
    edges: List[Edge] = []
    for block_idx in range(model.cfg.n_layers):
        attn_srcs = []
        for head_idx in range(model.cfg.n_heads):
            attn_src = EdgeSrc(
                name=f"A{block_idx}.{head_idx}",
                module_name=f"blocks.{block_idx}.attn.hook_result",
                _t_idx=(None, None, head_idx),
                weight=f"blocks.{block_idx}.attn.W_O",
                _weight_t_idx=head_idx,
            )
            attn_srcs.append(attn_src)
        resid_mid_dest = EdgeDest(
            name=f"Block {block_idx} Resid Mid",
            module_name=f"blocks.{block_idx}.hook_resid_mid",
            _t_idx=None,
            weight=None,
            _weight_t_idx=None,
        )
        for attn_src in attn_srcs:
            edges.append(Edge(src=attn_src, dest=resid_mid_dest))

        mlp_src = EdgeSrc(
            name=f"MLP {block_idx}",
            module_name=f"blocks.{block_idx}.mlp",
            _t_idx=None,
            weight=f"blocks.{block_idx}.mlp.W_out",
            _weight_t_idx=None,
        )
        last_block = block_idx + 1 == model.cfg.n_layers
        resid_post_dest = EdgeDest(
            name="Resid Final" if last_block else f"Block {block_idx} Resid Post",
            module_name=f"blocks.{block_idx}.hook_resid_post",
            _t_idx=None,
            weight=None,
            _weight_t_idx=None,
        )
        edges.append(Edge(src=mlp_src, dest=resid_post_dest))
    return OrderedSet(edges)
