from typing import List, Literal, Optional, Set

import torch as t

from auto_circuit.data import PromptDataLoader
from auto_circuit.prune_algos.subnetwork_probing import (
    SP_FAITHFULNESS_TARGET,
    init_mask_val,
    subnetwork_probing_prune_scores,
)
from auto_circuit.types import Edge, MaskFn, PruneScores
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel
from auto_circuit.utils.tensor_ops import prune_scores_threshold


def circuit_probing_prune_scores(
    model: PatchableModel,
    dataloader: PromptDataLoader,
    official_edges: Optional[Set[Edge]],
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
    faithfulness_target: SP_FAITHFULNESS_TARGET = "kl_div",
    validation_dataloader: Optional[PromptDataLoader] = None,
) -> PruneScores:
    """
    Wrapper of
    [Subnetwork Probing][auto_circuit.prune_algos.subnetwork_probing.subnetwork_probing_prune_scores]
    that searches for circuits of different sizes and assigns scores to the edges
    according to the size of the smallest circuit that they are part of. Smaller
    circuits have higher scores because they contain more important edges. Edges not in
    any circuit are assigned a score of `0`.

    Args:
        circuit_sizes: List of circuit sizes to probe. If `"true_size"` is in the list,
            then we include the size of `official_edges` in the list. If
            `official_edges` is `None`, then we raise an error.
    """  # noqa: W505, E501

    sizes = []
    for size in circuit_sizes:
        if size == "true_size":
            assert official_edges is not None
            size = len(official_edges)
        assert size > 0
        sizes.append(size)
    assert len(set(sizes)) == len(sizes)
    assert len(sizes) == len(circuit_sizes)
    sorted_circuit_sizes = sorted(sizes)

    prune_scores = model.new_prune_scores()

    # Iterate over the circuit sizes in ascending order
    for size_idx, size in enumerate((size_pbar := tqdm(sorted_circuit_sizes))):
        size_pbar.set_description(f"Circuit Probing Size {size}")
        assert (isinstance(size, int) and size > 0) or size is None
        new_prune_scores: PruneScores = subnetwork_probing_prune_scores(
            model=model,
            dataloader=dataloader,
            official_edges=official_edges,
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
            faithfulness_target=faithfulness_target,
            validation_dataloader=validation_dataloader,
        )
        assert all([t.all(ps >= 0) for ps in new_prune_scores.values()])
        threshold = prune_scores_threshold(new_prune_scores, size)
        score = len(sorted_circuit_sizes) - size_idx
        for mod, new_ps in new_prune_scores.items():
            curr_ps = prune_scores[mod]
            # Smaller circuits have higher scores. Bigger circuits don't overwrite
            new_circuit = (new_ps >= threshold) & (curr_ps == 0)
            prune_scores[mod] = t.where(new_circuit, score, curr_ps)
    return prune_scores
