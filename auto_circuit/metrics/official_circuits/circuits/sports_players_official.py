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
    The full Sports Players circuit from input to output, as discovered by
    [Rajamanoharan et al. (2023)](https://www.alignmentforum.org/posts/3tqJ65kuTkBh8wrRH/).

    Read the source code comments for precise details on our interpretation of the
    circuit. The focus of the paper was on probing for sports features, so the exact
    set of edges that constitute the circuit is somewhat ambiguous.

    Args:
        model: A patchable TransformerLens Pythia 2.8B `HookedTransformer` model.
        token_positions: Whether to distinguish between token positions when returning
            the set of circuit edges. If `True`, we require that the `model` has
            `seq_len` not `None` (ie. separate edges for each token position) and that
            `word_idxs` is provided.
        word_idxs: A dictionary defining the index of specific named tokens in the
            circuit definition. For this circuit, the required tokens positions are:
            <ul>
                <li><code>first_name_tok</code></li>
                <li><code>final_name_tok</code></li>
                <li><code>end</code></li>
            </ul>
        seq_start_idx: Offset to add to all of the token positions in `word_idxs`.
            This is useful when using KV caching to skip the common prefix of the
            prompt.

    Returns:
        The set of edges in the circuit.
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
    Wrapper for
    [`sports_players_true_edges`][auto_circuit.metrics.official_circuits.circuits.sports_players_official.sports_players_true_edges]
    that does not include the `extract_sport` section of the circuit. Instead, we extend
    the `lookup` MLP stack to the final layer of the model.

    This is included to make it easier to reproduce the probing results from the post
    which just probe the MLP stack and ignore the `extract_sport` section.
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
