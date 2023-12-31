# Based on:
# https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/ioi/utils.py
# I added the token positions based on the findings in the paper
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


def ioi_true_edges(model: PatchableModel, token_positions: bool = False) -> Set[Edge]:
    assert model.cfg.model_name == "gpt2"

    special_connections: List[Conn] = [
        Conn("INPUT", "previous token", [("q", 5), ("k", 4), ("v", 4)]),
        Conn("INPUT", "duplicate token", [("q", 10), ("k", 4), ("v", 4)]),
        Conn("INPUT", "s2 inhibition", [("q", 14)]),
        Conn("INPUT", "negative", [("k", 2), ("v", 2)]),
        Conn("INPUT", "name mover", [("k", 2), ("v", 2)]),
        Conn("INPUT", "backup name mover", [("k", 2), ("v", 2)]),
        Conn("previous token", "induction", [("k", 5), ("v", 5)]),
        Conn("induction", "s2 inhibition", [("k", 10), ("v", 10)]),
        Conn("duplicate token", "s2 inhibition", [("k", 10), ("v", 10)]),
        Conn("s2 inhibition", "negative", [("q", 14)]),
        Conn("s2 inhibition", "name mover", [("q", 14)]),
        Conn("s2 inhibition", "backup name mover", [("q", 14)]),
        Conn("negative", "OUTPUT", [(None, 14)]),
        Conn("name mover", "OUTPUT", [(None, 14)]),
        Conn("backup name mover", "OUTPUT", [(None, 14)]),
    ]
    edges_present: Dict[str, int] = {}
    for conn in special_connections:
        edge_src_names, edge_dests = [], []
        if conn.inp == "INPUT":
            edge_src_names = ["Resid Start"]
        else:
            for (layer, head) in IOI_CIRCUIT[conn.inp]:
                edge_src_names.append(f"A{layer}.{head}")
        if conn.out == "OUTPUT":
            assert len(conn.qkv) == 1
            final_tok_idx = conn.qkv[0][1]
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
                edges_present[f"{src_name}->{dest_name}"] = tok_pos

    true_edges: Set[Edge] = set()
    for edge in model.edges:
        if edge.name in edges_present.keys():
            if token_positions:
                true_edges.add(Edge(edge.src, edge.dest, edges_present[edge.name]))
            else:
                true_edges.add(edge)
    return true_edges
