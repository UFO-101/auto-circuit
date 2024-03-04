# Based on:
# https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/ioi/utils.py
# I added the token positions based on the findings in the paper
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from auto_circuit.types import Edge
from auto_circuit.utils.patchable_model import PatchableModel

IOI_CIRCUIT = {
    "name mover": [
        (9, 9),  # by importance
        (10, 0),
        (9, 6),
    ],
    "backup name mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative": [(10, 7), (11, 10)],
    "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate token": [
        (0, 1),
        (0, 10),
        (3, 0),
        # (7, 1),
    ],  # unclear exactly what (7,1) does
    "previous token": [
        (2, 2),
        # (2, 9),
        (4, 11),
        # (4, 3),
        # (4, 7),
        # (5, 6),
        # (3, 3),
        # (3, 7),
        # (3, 6),
    ],
}


@dataclass(frozen=True)
class Conn:
    inp: str
    out: str
    qkv: List[Tuple[Optional[str], int]]


def ioi_true_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    The edge-level circuit from the IOI paper.
    """
    assert model.cfg.model_name == "gpt2"
    assert model.is_factorized

    final_tok_idx = word_idxs.get("end", 0)
    io_tok_idx = word_idxs.get("IO", 0)
    s1_tok_idx = word_idxs.get("S1", 0)
    s2_tok_idx = word_idxs.get("S2", 0)
    s1_plus_1_tok_idx = word_idxs.get("S1+1", 0)

    if token_positions:
        assert final_tok_idx > 0, "word_idxs can't be empty if token_positions is True"

    special_connections: List[Conn] = [
        Conn(
            "INPUT",
            "previous token",
            [("q", s1_plus_1_tok_idx), ("k", s1_tok_idx), ("v", s1_tok_idx)],
        ),
        Conn(
            "INPUT",
            "duplicate token",
            [("q", s2_tok_idx), ("k", s1_tok_idx), ("v", s1_tok_idx)],
        ),
        Conn("INPUT", "s2 inhibition", [("q", final_tok_idx)]),
        Conn("INPUT", "negative", [("k", io_tok_idx), ("v", io_tok_idx)]),
        Conn("INPUT", "name mover", [("k", io_tok_idx), ("v", io_tok_idx)]),
        Conn("INPUT", "backup name mover", [("k", io_tok_idx), ("v", io_tok_idx)]),
        Conn(
            "previous token",
            "induction",
            [("k", s1_plus_1_tok_idx), ("v", s1_plus_1_tok_idx)],
        ),
        Conn("induction", "s2 inhibition", [("k", s2_tok_idx), ("v", s2_tok_idx)]),
        Conn(
            "duplicate token", "s2 inhibition", [("k", s2_tok_idx), ("v", s2_tok_idx)]
        ),
        Conn("s2 inhibition", "negative", [("q", final_tok_idx)]),
        Conn("s2 inhibition", "name mover", [("q", final_tok_idx)]),
        Conn("s2 inhibition", "backup name mover", [("q", final_tok_idx)]),
        Conn("negative", "OUTPUT", [(None, final_tok_idx)]),
        Conn("name mover", "OUTPUT", [(None, final_tok_idx)]),
        Conn("backup name mover", "OUTPUT", [(None, final_tok_idx)]),
    ]

    edges_present: Dict[str, List[int]] = defaultdict(list)
    for conn in special_connections:
        edge_src_names, edge_dests = [], []
        if conn.inp == "INPUT":
            edge_src_names = ["Resid Start"]
        else:
            for (layer, head) in IOI_CIRCUIT[conn.inp]:
                edge_src_names.append(f"A{layer}.{head}")
        if conn.out == "OUTPUT":
            assert len(conn.qkv) == 1
            edge_dests.append(("Resid End", final_tok_idx))
        else:
            for (layer, head) in IOI_CIRCUIT[conn.out]:
                for qkv in conn.qkv:
                    assert qkv[0] is not None
                    edge_dests.append((f"A{layer}.{head}.{qkv[0].upper()}", qkv[1]))

        # Connect all MLPS in between heads in the circuit
        # (in the IOI paper they allow activations to flow through MLPs,
        # which is equalivent to including all MLPs in between two nodes.)
        if conn.inp == "INPUT":
            src_layer = 0
        else:
            src_layer = min([layer for (layer, _) in IOI_CIRCUIT[conn.inp]])

        if conn.out == "OUTPUT":
            dest_layer = conn.qkv[0][1]
        else:
            dest_layer = max([layer for (layer, _) in IOI_CIRCUIT[conn.out]])
        dest_tok_idxs = [tok_idx for (_, tok_idx) in conn.qkv]

        # Src layer is inclusive because MLP comes after ATTN
        for layer in range(src_layer, dest_layer):
            for tok_idx in dest_tok_idxs:
                edge_src_names.append(f"MLP {layer}")
                edge_dests.append((f"MLP {layer}", tok_idx))

        for src_name in edge_src_names:
            for dest_name, tok_pos in edge_dests:
                edges_present[f"{src_name}->{dest_name}"].append(tok_pos)

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


def ioi_head_based_official_edges(
    model: PatchableModel,
    token_positions: bool = False,
    word_idxs: Dict[str, int] = {},
    seq_start_idx: int = 0,
) -> Set[Edge]:
    """
    Works for both factorized and unfactorized models.
    """
    assert model.cfg.model_name == "gpt2"
    assert (model.seq_len is not None) == token_positions

    final_tok_idx = word_idxs.get("end", 0)
    S2_tok_idx = word_idxs.get("S2", 0)
    S1_plus_1_tok_idx = word_idxs.get("S1+1", 0)

    if token_positions:
        assert final_tok_idx > 0, "word_idxs can't be empty if token_positions is True"

    CIRCUIT = {
        "name mover": [(9, 9), (10, 0), (9, 6)],
        "backup name mover": [
            (10, 10),
            (10, 6),
            (10, 2),
            (10, 1),
            (11, 2),
            (9, 7),
            (9, 0),
            (11, 9),
        ],
        "negative name mover": [(10, 7), (11, 10)],
        "s2 inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
        "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
        "duplicate token": [(0, 1), (0, 10), (3, 0)],
        "previous token": [(2, 2), (4, 11)],
    }

    SEQ_POS_TO_KEEP = {
        "name mover": final_tok_idx,
        "backup name mover": final_tok_idx,
        "negative name mover": final_tok_idx,
        "s2 inhibition": final_tok_idx,
        "induction": S2_tok_idx,
        "duplicate token": S2_tok_idx,
        "previous token": S1_plus_1_tok_idx,
    }
    heads_to_keep: Set[Tuple[str, Optional[int]]] = set()
    for head_type, head_idxs in CIRCUIT.items():
        head_type_seq_idx = SEQ_POS_TO_KEEP[head_type]
        for head_lyr, head_idx in head_idxs:
            head_name = f"A{head_lyr}.{head_idx}"
            if token_positions:
                heads_to_keep.add((head_name, head_type_seq_idx))
            else:
                heads_to_keep.add((head_name, None))

    official_edges: Set[Edge] = set()
    not_official_edges: Set[Edge] = set()
    for edge in model.edges:
        src_is_head = edge.src.head_idx is not None
        if edge.seq_idx is not None:
            assert token_positions
            src_head_key = (edge.src.name, edge.seq_idx + seq_start_idx)
        else:
            assert not token_positions
            src_head_key = (edge.src.name, None)

        if src_is_head and src_head_key not in heads_to_keep:
            not_official_edges.add(edge)
            continue

        official_edges.add(edge)
    return official_edges
