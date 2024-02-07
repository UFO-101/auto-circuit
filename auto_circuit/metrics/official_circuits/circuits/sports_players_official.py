# Based on:
# https://www.lesswrong.com/posts/3tqJ65kuTkBh8wrRH/fact-finding-simplifying-the-circuit-post-2#Token_concatenation_is_achieved_by_attention_heads_moving_token_embeddings  # noqa: E501
from typing import Dict, List, Set

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel


def sports_players_true_edges(
    model: PatchableModel, token_positions: bool = False, seq_start_idx: int = 0
) -> Set[Edge]:
    """
    !!! Note: !!!
    The sequence positions assume prompts of length 20 (3 name tokens), as in
    sports_players_pythia-2.8b-deduped_prompts.json
    !!! /Note !!!

    This is the sequence length for which pythia-2.8b-deduped assigns the most correct
    answers > 0.5 probability. (See datasets/sports-players/sports_players_generator.py)
    """
    assert model.cfg.model_name == "pythia-2.8b-deduped"
    first_name_tok_idx = 13
    final_name_tok_idx = 15
    non_final_name_toks_idxs = list(range(first_name_tok_idx, final_name_tok_idx))
    last_tok_idx = 19

    edges_present: Dict[str, List[int]] = {}

    # --- Concaternate Tokens and Lookup ---
    # We include every edge in the first 2 layers at the first two name tokens, since
    # DeepMind didn't investigate the case of 3 token embeddings with the 1 shot prompt
    # in detail.
    n_heads = model.cfg.n_heads
    head_outs_0 = [f"A0.{head}" for head in range(n_heads)]
    head_outs_01 = [f"A{layer}.{head}" for layer in range(2) for head in range(n_heads)]

    # Many of these are not real edges. We filter them out at the end of this function.
    non_final_name_tok_nodes = ["Resid Start", "MLP 0"] + head_outs_0
    for src_node in non_final_name_tok_nodes:
        for dest_node in non_final_name_tok_nodes:
            edges_present[f"{src_node}->{dest_node}"] = non_final_name_toks_idxs

    # We include every edge in the first 2 layers and every MLP from layers 2-8 at the
    # last name token
    mlps = [f"MLP {layer}" for layer in range(0, 9)]
    final_name_tok_nodes = ["Resid Start"] + head_outs_01 + mlps
    for src_node in final_name_tok_nodes:
        for dest_node in final_name_tok_nodes:
            edges_present[f"{src_node}->{dest_node}"] = [final_name_tok_idx]

    # --- Extract sport ---
    # Lookup to main attention head
    main_attn_head = "A16.20"
    for src_node in final_name_tok_nodes:
        edges_present[f"{src_node}->{main_attn_head}"] = [final_name_tok_idx]

    # V-composition from A16.20 to the other important attention heads
    secondary_attn_heads = ["A21.9", "A22.17", "A22.15", "A17.30", "A19.24"]
    for attn_head in secondary_attn_heads:
        edges_present[f"{main_attn_head}->{attn_head}"] = [final_name_tok_idx]

    # Attention heads to Resid End
    for attn_head in [main_attn_head] + secondary_attn_heads:
        edges_present[f"{attn_head}->Resid End"] = [last_tok_idx]

    true_edges: Set[Edge] = set()
    for edge in model.edges:
        if edge.name in edges_present.keys():
            if token_positions:
                for tok_pos in edges_present[edge.name]:
                    true_edges.add(Edge(edge.src, edge.dest, tok_pos - seq_start_idx))
            else:
                true_edges.add(edge)
    return true_edges
