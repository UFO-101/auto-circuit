from itertools import count
from typing import Set, Tuple

import transformer_lens as tl

from auto_circuit.types import DestNode, SrcNode


def factorized_src_nodes(model: tl.HookedTransformer) -> Set[SrcNode]:
    """Get the source part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    assert model.cfg.use_attn_result  # Get attention head outputs separately
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers, idxs = count(), count()
    nodes = set()
    nodes.add(
        SrcNode(
            name="Resid Start",
            module_name="blocks.0.hook_resid_pre",
            layer=next(layers),
            idx=next(idxs),
            weight="embed.W_E",
        )
    )

    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        for head_idx in range(model.cfg.n_heads):
            nodes.add(
                SrcNode(
                    name=f"A{block_idx}.{head_idx}",
                    module_name=f"blocks.{block_idx}.attn.hook_result",
                    layer=layer,
                    idx=next(idxs),
                    head_dim=2,
                    head_idx=head_idx,
                    weight=f"blocks.{block_idx}.attn.W_O",
                    weight_head_dim=0,
                )
            )
        if not model.cfg.attn_only:
            nodes.add(
                SrcNode(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.hook_mlp_out",
                    layer=layer if model.cfg.parallel_attn_mlp else next(layers),
                    idx=next(idxs),
                    weight=f"blocks.{block_idx}.mlp.W_out",
                )
            )
    return nodes


def factorized_dest_nodes(
    model: tl.HookedTransformer, separate_qkv: bool
) -> Set[DestNode]:
    """Get the destination part of each edge in the factorized graph, grouped by layer.
    Graph is factorized following the Mathematical Framework paper."""
    if separate_qkv:
        assert model.cfg.use_split_qkv_input  # Separate Q, K, V input for each head
    else:
        assert model.cfg.use_attn_in
    assert model.cfg.use_hook_mlp_in  # Get MLP input BEFORE layernorm
    layers, idxs = count(1), count()
    nodes = set()
    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        for head_idx in range(model.cfg.n_heads):
            if separate_qkv:
                for letter in ["Q", "K", "V"]:
                    nodes.add(
                        DestNode(
                            name=f"A{block_idx}.{head_idx}.{letter}",
                            module_name=f"blocks.{block_idx}.hook_{letter.lower()}_input",
                            layer=layer,
                            idx=next(idxs),
                            head_dim=2,
                            head_idx=head_idx,
                            weight=f"blocks.{block_idx}.attn.W_{letter}",
                            weight_head_dim=0,
                        )
                    )
            else:
                nodes.add(
                    DestNode(
                        name=f"A{block_idx}.{head_idx}",
                        module_name=f"blocks.{block_idx}.hook_attn_in",
                        layer=layer,
                        idx=next(idxs),
                        head_dim=2,
                        head_idx=head_idx,
                        weight=f"blocks.{block_idx}.attn.W_QKV",
                        weight_head_dim=0,
                    )
                )
        if not model.cfg.attn_only:
            nodes.add(
                DestNode(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.hook_mlp_in",
                    layer=layer if model.cfg.parallel_attn_mlp else next(layers),
                    idx=next(idxs),
                    weight=f"blocks.{block_idx}.mlp.W_in",
                )
            )
    nodes.add(
        DestNode(
            name="Resid End",
            module_name=f"blocks.{model.cfg.n_layers - 1}.hook_resid_post",
            layer=next(layers),
            idx=next(idxs),
            weight="unembed.W_U",
        )
    )
    return nodes


def simple_graph_nodes(
    model: tl.HookedTransformer,
) -> Tuple[Set[SrcNode], Set[DestNode]]:
    """Get the nodes in the unfactorized graph."""
    assert not model.cfg.parallel_attn_mlp
    layers, src_idxs, dest_idxs = count(), count(), count()
    src_nodes, dest_nodes = set(), set()
    for block_idx in range(model.cfg.n_layers):
        layer = next(layers)
        for head_idx in range(model.cfg.n_heads):
            src_nodes.add(
                SrcNode(
                    name=f"A{block_idx}.{head_idx}",
                    module_name=f"blocks.{block_idx}.attn.hook_result",
                    layer=layer,
                    idx=next(src_idxs),
                    head_idx=head_idx,
                    head_dim=2,
                    weight=f"blocks.{block_idx}.attn.W_O",
                    weight_head_dim=0,
                )
            )
        dest_nodes.add(
            DestNode(
                name=f"Block {block_idx} Resid Mid",
                module_name=f"blocks.{block_idx}.hook_resid_mid",
                layer=next(layers),
                idx=next(dest_idxs),
            )
        )
        if not model.cfg.attn_only:
            src_nodes.add(
                SrcNode(
                    name=f"MLP {block_idx}",
                    module_name=f"blocks.{block_idx}.mlp",
                    layer=next(layers),
                    idx=next(src_idxs),
                    weight=f"blocks.{block_idx}.mlp.W_out",
                )
            )
        last_block = block_idx + 1 == model.cfg.n_layers
        dest_nodes.add(
            DestNode(
                name="Resid Final" if last_block else f"Block {block_idx} Resid Post",
                module_name=f"blocks.{block_idx}.hook_resid_post",
                layer=next(layers),
                idx=next(dest_idxs),
            )
        )
    return src_nodes, dest_nodes
