from typing import Dict, List, Optional, Set

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel

CIRCUIT = {
    # "input": [None], # special case input
    "ATTN_0.3_0.5": [(0, 3), (0, 5)],
    "ATTN_0.1": [(0, 1)],
    "EARLY_MLPS": [(0, None), (1, None), (2, None), (3, None)],
    "MID_ATTNS": [(5, 5), (6, 1), (6, 9), (7, 10), (8, 11), (9, 1)],
    "LATE_MLPS": [(8, None), (9, None), (10, None), (11, None)],
    # output special case
}

CONNECTED_PAIRS = [
    ("ATTN_0.1", "EARLY_MLPS"),
    ("ATTN_0.1", "MID_ATTNS"),
    ("ATTN_0.3_0.5", "MID_ATTNS"),
    ("EARLY_MLPS", "MID_ATTNS"),
    ("MID_ATTNS", "LATE_MLPS"),
]


def idx_to_nodes(layer_idx: int, head_idx: Optional[int], src_nodes: bool) -> List[str]:
    if src_nodes:
        if head_idx is None:
            return [f"MLP {layer_idx}"]
        else:
            return [f"A{layer_idx}.{head_idx}"]
    else:
        if head_idx is None:
            return [f"MLP {layer_idx}"]
        else:
            return [(f"A{layer_idx}.{head_idx}.{letter}") for letter in "QKV"]


def greaterthan_true_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The Greaterthan circuit, discovered by
    [Hanna et al. 2023](https://arxiv.org/abs/2305.00586).

    Based on the [ACDC implementation](https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/greaterthan/utils.py).

    Args:
        model: A patchable TransformerLens GPT-2 `HookedTransformer` model.
        token_positions: Whether to distinguish between token positions when returning
            the set of circuit edges. If `True`, we require that the `model` has
            `seq_len` not `None` (ie. separate edges for each token position) and that
            `word_idxs` is provided.
        word_idxs: A dictionary defining the index of specific named tokens in the
            circuit definition. For this circuit, token positions are not specified, so
            this parameter is not used.
        seq_start_idx: Offset to add to all of the token positions in `word_idxs`.
            For this circuit, token positions are not specified, so this parameter is
            not used.

    Returns:
        The set of edges in the circuit.

    Note:
        The Greaterthan circuit does not specify token positions, so if
        `token_positions` is `True`, all token positions are included for the edges in
        the circuit.
    """
    assert model.cfg.model_name == "gpt2"
    assert model.separate_qkv

    edges_present: List[str] = []

    # attach input
    for GROUP in ["ATTN_0.3_0.5", "ATTN_0.1", "EARLY_MLPS"]:
        for layer_idx, head_idx in CIRCUIT[GROUP]:
            dest_nodes = idx_to_nodes(layer_idx, head_idx, src_nodes=False)

            for node_name in dest_nodes:
                edges_present.append(f"Resid Start->{node_name}")

    # attach output
    for GROUP in ["MID_ATTNS", "LATE_MLPS"]:
        for layer_idx, head_idx in CIRCUIT[GROUP]:
            src_nodes = idx_to_nodes(layer_idx, head_idx, src_nodes=True)
            for node_name in src_nodes:
                edges_present.append(f"{node_name}->Resid End")

    # MLP groups are interconnected
    for GROUP in ["EARLY_MLPS", "LATE_MLPS"]:
        for src_layer, _ in CIRCUIT[GROUP]:
            for dest_layer, _ in CIRCUIT[GROUP]:
                if src_layer >= dest_layer:
                    continue
                edges_present.append(f"MLP {src_layer}->MLP {dest_layer}")

    # connected pairs
    for GROUP1, GROUP2 in CONNECTED_PAIRS:
        for src_layer, src_head in CIRCUIT[GROUP1]:
            for dest_layer, dest_head in CIRCUIT[GROUP2]:
                src_is_attn = src_head is not None
                dest_is_mlp = dest_head is None
                same_layer = src_layer == dest_layer
                attn_to_mlp_on_same_layer = src_is_attn and dest_is_mlp and same_layer
                if src_layer >= dest_layer and not attn_to_mlp_on_same_layer:
                    continue
                for src_node_name in idx_to_nodes(src_layer, src_head, src_nodes=True):
                    for dest_node_name in idx_to_nodes(
                        dest_layer, dest_head, src_nodes=False
                    ):
                        edges_present.append(f"{src_node_name}->{dest_node_name}")

    # Hanna et al have totally clean query inputs to MID_ATTNS heads
    # this is A LOT of edges so we just add the MLP -> MID_ATTNS Q edges

    MAX_AMID_LAYER = max([layer_idx for layer_idx, _ in CIRCUIT["MID_ATTNS"]])
    # connect all MLPs before the MID_ATTNS heads
    for mlp_sender_layer in range(0, MAX_AMID_LAYER):
        for mlp_receiver_layer in range(1 + mlp_sender_layer, MAX_AMID_LAYER):
            edges_present.append(f"MLP {mlp_sender_layer}->MLP {mlp_receiver_layer}")

    # connect all early MLPs to MID_ATTNS heads' Q inputs
    for layer_idx, head_idx in CIRCUIT["MID_ATTNS"]:
        for mlp_sender_layer in range(0, layer_idx):
            edges_present.append(f"MLP {mlp_sender_layer}->A{layer_idx}.{head_idx}.Q")

    true_edges: Set[Edge] = set()
    for edge in model.edges:
        if edge.name in edges_present:
            true_edges.add(edge)
    return true_edges
