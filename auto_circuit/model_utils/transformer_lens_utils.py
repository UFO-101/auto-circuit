from typing import List

import torch as t
import transformer_lens as tl
from ordered_set import OrderedSet

from auto_circuit.types import Edge, EdgeDest, EdgeSrc


def factorized_graph_src_layers(
    model: tl.HookedTransformer,
) -> List[OrderedSet[EdgeSrc]]:
    """Get the source part of each edge, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert model.cfg.use_split_qkv_input  # Get Q, K, V for each head separately
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers = []
    assert isinstance(model.blocks[0].hook_resid_pre, t.nn.Module)
    resid_start = EdgeSrc(
        name="Resid Start",
        module=model.blocks[0].hook_resid_pre,
        _t_idx=None,
        weight="embed.W_E",
        _weight_t_idx=None,
    )
    layers.append(OrderedSet([resid_start]))
    for block_idx, block in enumerate(model.blocks):
        attn_set = OrderedSet([])
        for head_idx in range(model.cfg.n_heads):
            attn_src = EdgeSrc(
                name=f"A{block_idx}.{head_idx}",
                module=block.attn.hook_result,  # type: ignore
                _t_idx=(None, None, head_idx),
                weight=f"blocks.{block_idx}.attn.W_O",
                _weight_t_idx=head_idx,
            )
            attn_set.add(attn_src)
        layers.append(attn_set)
        assert isinstance(block.mlp, t.nn.Module)
        mlp_set = OrderedSet(
            [
                EdgeSrc(
                    name=f"MLP {block_idx}",
                    module=block.mlp,
                    _t_idx=None,
                    weight=f"blocks.{block_idx}.mlp.W_out",
                    _weight_t_idx=None,
                )
            ]
        )
        layers.append(mlp_set)
    return layers


def factorized_graph_dest_layers(
    model: tl.HookedTransformer,
) -> List[OrderedSet[EdgeDest]]:
    """Get the destination part of each edge, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert model.cfg.use_split_qkv_input  # Get Q, K, V for each head separately
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers = []
    for block_idx, block in enumerate(model.blocks):
        attn_set = OrderedSet([])
        for head_idx in range(model.cfg.n_heads):
            for letter, input in [
                ("Q", block.hook_q_input),
                ("K", block.hook_k_input),
                ("V", block.hook_v_input),
            ]:
                assert isinstance(input, t.nn.Module) and isinstance(input.name, str)
                attn_dest = EdgeDest(
                    name=f"A{block_idx}.{head_idx}.{letter}",
                    module=input,
                    _t_idx=(None, None, head_idx),
                    weight=f"blocks.{block_idx}.attn.W_{letter}",
                    _weight_t_idx=head_idx,
                )
                attn_set.add(attn_dest)
        layers.append(attn_set)
        assert isinstance(block.hook_mlp_in, t.nn.Module)
        mlp_dest = EdgeDest(
            name=f"MLP {block_idx}",
            module=block.hook_mlp_in,
            _t_idx=None,
            weight=f"blocks.{block_idx}.mlp.W_in",
            _weight_t_idx=None,
        )
        layers.append(set([mlp_dest]))
    assert isinstance(model.blocks[-1].hook_resid_post, t.nn.Module)
    resid_end = EdgeDest(
        name="Resid End",
        module=model.blocks[-1].hook_resid_post,
        _t_idx=None,
        weight="unembed.W_U",
        _weight_t_idx=None,
    )
    layers.append(set([resid_end]))
    return layers


def simple_graph_edges(model: tl.HookedTransformer) -> OrderedSet[Edge]:
    edges: List[Edge] = []
    prev_resid_src: EdgeSrc | None = None
    for block_idx, block in enumerate(model.blocks):
        assert isinstance(block.hook_resid_pre, t.nn.Module)
        if prev_resid_src is None:
            prev_resid_src = EdgeSrc(
                name="Embed",
                module=block.hook_resid_pre,
                _t_idx=None,
                weight="embed.W_E" if block_idx == 0 else None,
                _weight_t_idx=None,
            )
        attn_srcs = []
        attn_dests = []
        for head_idx in range(model.cfg.n_heads):
            attn_dest = EdgeDest(
                name=f"A{block_idx}.{head_idx}",
                module=block.attn.hook_result,  # type: ignore
                _t_idx=(None, None, head_idx),
                weight=f"blocks.{block_idx}.attn.W_O",
                _weight_t_idx=head_idx,
            )
            attn_dests.append(attn_dest)
            attn_src = EdgeSrc(
                name=f"A{block_idx}.{head_idx}",
                module=block.attn.hook_result,  # type: ignore
                _t_idx=(None, None, head_idx),
                weight=f"blocks.{block_idx}.attn.W_O",
                _weight_t_idx=head_idx,
            )
            attn_srcs.append(attn_src)
        assert isinstance(block.hook_resid_mid, t.nn.Module)
        resid_mid_dest = EdgeDest(
            name=f"Block {block_idx} Resid Mid",
            module=block.hook_resid_mid,
            _t_idx=None,
            weight=None,
            _weight_t_idx=None,
        )
        edges.append(Edge(src=prev_resid_src, dest=resid_mid_dest))
        for attn_src, attn_dest in zip(attn_srcs, attn_dests):
            edges.append(Edge(src=prev_resid_src, dest=attn_dest))
            edges.append(Edge(src=attn_src, dest=resid_mid_dest))

        resid_mid_src = EdgeSrc(
            name=f"Block {block_idx} Resid Mid",
            module=block.hook_resid_mid,
            _t_idx=None,
            weight=None,
            _weight_t_idx=None,
        )
        assert isinstance(block.mlp, t.nn.Module)
        mlp_src = EdgeSrc(
            name=f"MLP {block_idx}",
            module=block.mlp,
            _t_idx=None,
            weight=f"blocks.{block_idx}.mlp.W_out",
            _weight_t_idx=None,
        )
        mlp_dest = EdgeDest(
            name=f"MLP {block_idx}",
            module=block.mlp,
            _t_idx=None,
            weight=f"blocks.{block_idx}.mlp.W_out",
            _weight_t_idx=None,
        )
        assert isinstance(block.hook_resid_post, t.nn.Module)
        last_block = block_idx + 1 == model.cfg.n_layers
        resid_post_dest = EdgeDest(
            name="Unembed" if last_block else f"Block {block_idx} Resid Post",
            module=block.hook_resid_post,
            _t_idx=None,
            weight="unembed.W_U" if last_block else None,
            _weight_t_idx=None,
        )
        resid_post_src = EdgeSrc(
            name="Unembed" if last_block else f"Block {block_idx} Resid Post",
            module=block.hook_resid_post,
            _t_idx=None,
            weight="unembed.W_U" if last_block else None,
            _weight_t_idx=None,
        )
        prev_resid_src = resid_post_src
        edges.append(Edge(src=resid_mid_src, dest=resid_post_dest))
        edges.append(Edge(src=resid_mid_src, dest=mlp_dest))
        edges.append(Edge(src=mlp_src, dest=resid_post_dest))
    return OrderedSet(edges)
