from dataclasses import dataclass
import torch as t
from typing import Dict

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge
from torch.utils.data import DataLoader

from auto_circuit.utils.graph_utils import graph_edges


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
    qkv: tuple[str, ...]

def ioi_true_edges_prune_scores(
    model: t.nn.Module,
    factorized: bool,
    train_data: DataLoader[PromptPairBatch],
) -> Dict[Edge, float]:
    special_connections: set[Conn] = {
        Conn("INPUT", "previous token", ("q", "k", "v")),
        Conn("INPUT", "duplicate token", ("q", "k", "v")),
        Conn("INPUT", "s2 inhibition", ("q",)),
        Conn("INPUT", "negative", ("k", "v")),
        Conn("INPUT", "name mover", ("k", "v")),
        Conn("INPUT", "backup name mover", ("k", "v")),
        Conn("previous token", "induction", ("k", "v")),
        Conn("induction", "s2 inhibition", ("k", "v")),
        Conn("duplicate token", "s2 inhibition", ("k", "v")),
        Conn("s2 inhibition", "negative", ("q",)),
        Conn("s2 inhibition", "name mover", ("q",)),
        Conn("s2 inhibition", "backup name mover", ("q",)),
        Conn("negative", "OUTPUT", ()),
        Conn("name mover", "OUTPUT", ()),
        Conn("backup name mover", "OUTPUT", ()),
    }
    edges_present: List[str] = []
    for conn in special_connections:
        edge_src_names = []
        if conn.inp == "INPUT":
            edge_src_names = ["Resid Start"]
        else:
            for (layer, head) in IOI_CIRCUIT[conn.inp]:
                edge_src_names.append(f"A.{layer}.{head}")
        edge_dest_names = []
        if conn.out == "OUTPUT":
            edge_dest_names = ["Resid End"]
        else:
            for (layer, head) in IOI_CIRCUIT[conn.out]:
                edge_dest_names.append(f"A.{layer}.{head}")
        
        for src_name in edge_src_names:
            for dest_name in edge_dest_names:
                edges_present.append(f"{src_name}->{dest_name}")


    edges = graph_edges(model, factorized)
    prune_scores = {}
    for edge in edges:
        if edge.name in edges_present:
            prune_scores[edge] = 1.0
        else:
            prune_scores[edge] = 0.0
    return prune_scores