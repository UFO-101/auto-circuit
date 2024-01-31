from typing import List, Literal, Optional, Set

from auto_circuit.prune_algos.subnetwork_probing import (
    init_mask_val,
    subnetwork_probing_prune_scores,
)
from auto_circuit.tasks import Task
from auto_circuit.types import Edge, PruneScores
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.tensor_ops import MaskFn


def circuit_probing_prune_scores(
    task: Task,
    learning_rate: float = 0.1,
    epochs: int = 20,
    regularize_lambda: float = 10,
    mask_fn: MaskFn = "hard_concrete",
    dropout_p: float = 0.0,
    init_val: float = -init_mask_val,
    show_train_graph: bool = False,
    circuit_sizes: List[int | Literal["true_size"]] = ["true_size"],
    tree_optimisation: bool = False,
    avoid_edges: Optional[Set[Edge]] = None,
    avoid_lambda: float = 1.0,
) -> PruneScores:

    sizes = []
    for size in circuit_sizes:
        if size == "true_size":
            assert task.true_edges is not None
            size = len(task.true_edges)
        assert isinstance(size, int) and size > 0
        sizes.append(size)
    assert len(set(sizes)) == len(sizes)
    assert len(sizes) == len(circuit_sizes)
    sorted_circuit_sizes = sorted(sizes)

    prune_scores: PruneScores = {}
    for size_idx, size in (size_pbar := tqdm(enumerate(sorted_circuit_sizes))):
        size_pbar.set_description(f"Circuit Probing Size {size}")
        assert (isinstance(size, int) and size > 0) or size is None
        new_prune_scores: PruneScores = subnetwork_probing_prune_scores(
            task=task,
            learning_rate=learning_rate,
            epochs=epochs,
            regularize_lambda=regularize_lambda,
            mask_fn=mask_fn,
            dropout_p=dropout_p,
            init_val=init_val,
            show_train_graph=show_train_graph,
            circuit_size=size,
            tree_optimisation=tree_optimisation,
            avoid_edges=avoid_edges,
            avoid_lambda=avoid_lambda,
        )
        for edge in new_prune_scores.keys():
            if edge not in prune_scores:
                prune_scores[edge] = len(sorted_circuit_sizes) - size_idx
    return prune_scores
