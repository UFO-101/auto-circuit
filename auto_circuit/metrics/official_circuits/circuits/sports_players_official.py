# Based on:
# https://www.lesswrong.com/posts/3tqJ65kuTkBh8wrRH/fact-finding-simplifying-the-circuit-post-2#Token_concatenation_is_achieved_by_attention_heads_moving_token_embeddings  # noqa: E501
from collections import defaultdict
from typing import Dict, List, Set

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel


def sports_players_true_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The full circuit for sports players from input to input for the full prompt.
    """
    assert model.cfg.model_name == "pythia-2.8b-deduped"
    assert model.separate_qkv is False, "Sports players doesn't support separate QKV"

    first_name_tok_idx = word_idxs.get("first_name_tok", 0)
    final_name_tok_idx = word_idxs.get("final_name_tok", 0)
    non_final_name_toks_idxs = list(range(first_name_tok_idx, final_name_tok_idx))
    final_tok_idx = word_idxs.get("end", 0)

    if token_positions:
        assert final_tok_idx > 0, "Must provide word_idxs if token_positions is True"

    # Many edge names in this dict will not exist. We filter them at the end.
    edges_present: Dict[str, List[int]] = defaultdict(list)

    # ------------------------- Concaternate Tokens and Lookup -------------------------
    # We include every edge between nodes in the first 2 layers at the first two (out of
    # three) name tokens, except MLP 1 since this can't affect the attention heads at
    # the final name token, where the final representation of the name is output to the
    # residual stream at the beginning of layer 2 (the third layer with 0 indexing).
    #
    # DeepMind did find a more detailed view of how the final resprentation is a linear
    # combination of the token embeddings (+MLP 0 embedding). But this was for the case
    # of 2 name tokens, whereas we are using 3 name tokens (because it provides more
    # data points) so we use this coarser view.
    n_heads = model.cfg.n_heads
    heads_01 = [f"A{layer}.{head}" for layer in range(2) for head in range(n_heads)]

    non_final_name_tok_nodes = ["Resid Start", "MLP 0"] + heads_01
    for src_node in non_final_name_tok_nodes:
        for dest_node in non_final_name_tok_nodes:
            edges_present[f"{src_node}->{dest_node}"].extend(non_final_name_toks_idxs)

    # We include every edge between [[all nodes in the first 2 layers] and [every MLP
    # from layers 2-8]] at the last name token.
    mlps = [f"MLP {layer}" for layer in range(0, 16)]
    final_name_tok_nodes = ["Resid Start"] + heads_01 + mlps
    for src_node in final_name_tok_nodes:
        for dest_node in final_name_tok_nodes:
            edges_present[f"{src_node}->{dest_node}"].append(final_name_tok_idx)

    # --------------------------------- Extract sport ----------------------------------
    # Lookup to main attention head
    main_attn_head = "A16.20"
    for src_node in final_name_tok_nodes:
        edges_present[f"{src_node}->{main_attn_head}"].append(final_name_tok_idx)

    # V-composition from A16.20 to the other important attention heads
    secondary_attn_heads = ["A21.9", "A22.17", "A22.15", "A17.30", "A19.24"]
    for attn_head in secondary_attn_heads:
        edges_present[f"{main_attn_head}->{attn_head}"].append(final_name_tok_idx)

    # Attention heads to Resid End
    for attn_head in [main_attn_head] + secondary_attn_heads:
        edges_present[f"{attn_head}->Resid End"].append(final_tok_idx)

    true_edges: Set[Edge] = set()
    for edge in model.edges:
        if edge.name in edges_present.keys():
            if token_positions:
                assert edge.seq_idx is not None
                if (edge.seq_idx + seq_start_idx) in edges_present[edge.name]:
                    true_edges.add(edge)
            else:
                true_edges.add(edge)
    return true_edges


def sports_players_probe_true_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The sports players circuit up to and including the lookup MLP stack extended to the
    final layer of the model. Intended to be used with the extract_sport probe.
    """
    assert model.cfg.model_name == "pythia-2.8b-deduped"
    assert model.separate_qkv is False, "Sports players doesn't support separate QKV"

    first_name_tok_idx = word_idxs.get("first_name_tok", 0)
    final_name_tok_idx = word_idxs.get("final_name_tok", 0)
    non_final_name_toks_idxs = list(range(first_name_tok_idx, final_name_tok_idx))
    final_tok_idx = word_idxs.get("end", 0)

    if token_positions:
        assert final_tok_idx > 0, "Must provide word_idxs if token_positions is True"

    # Many edge names in this dict will not exist. We filter them at the end.
    edges_present: Dict[str, List[int]] = defaultdict(list)

    # ------------------------- Concaternate Tokens and Lookup -------------------------
    # We include every edge between nodes in the first 2 layers at the first two (out of
    # three) name tokens, except MLP 1 since this can't affect the attention heads at
    # the final name token, where the final representation of the name is output to the
    # residual stream at the beginning of layer 2 (the third layer with 0 indexing).
    #
    # DeepMind did find a more detailed view of how the final resprentation is a linear
    # combination of the token embeddings (+MLP 0 embedding). But this was for the case
    # of 2 name tokens, whereas we are using 3 name tokens (because it provides more
    # data points) so we use this coarser view.
    n_heads = model.cfg.n_heads
    heads_01 = [f"A{layer}.{head}" for layer in range(2) for head in range(n_heads)]

    non_final_name_tok_nodes = ["Resid Start", "MLP 0"] + heads_01
    for src_node in non_final_name_tok_nodes:
        for dest_node in non_final_name_tok_nodes:
            edges_present[f"{src_node}->{dest_node}"].extend(non_final_name_toks_idxs)

    # We include every edge between [[all nodes in the first 2 layers] and [every MLP
    # from layers 2 onwards]] at the last name token.
    mlps = [f"MLP {layer}" for layer in range(0, model.cfg.n_layers)]
    final_name_tok_nodes = ["Resid Start"] + heads_01 + mlps
    for src_node in final_name_tok_nodes:
        for dest_node in final_name_tok_nodes:
            edges_present[f"{src_node}->{dest_node}"].append(final_name_tok_idx)

    true_edges: Set[Edge] = set()
    for edge in model.edges:
        if edge.name in edges_present.keys():
            if token_positions:
                assert edge.seq_idx is not None
                if (edge.seq_idx + seq_start_idx) in edges_present[edge.name]:
                    true_edges.add(edge)
            else:
                true_edges.add(edge)
    return true_edges
