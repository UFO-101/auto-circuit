from collections import defaultdict
from typing import Dict, List, Tuple

import torch as t
from ordered_set import OrderedSet
from torch.utils.data import DataLoader

from auto_circuit.data import PromptPairBatch
from auto_circuit.run_experiments import measure_kl_div, run_pruned
from auto_circuit.types import ActType, Edge, ExperimentType
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util, graph_edges


def acdc_prune_scores(
    model: t.nn.Module,
    factorized: bool,
    train_data: DataLoader[PromptPairBatch],
    tao_range: Tuple[float, float] = (0.1, 0.9),
    tao_step: float = 0.1,
) -> Dict[Edge, float]:
    """Run the ACDC algorithm from the paper 'Towards Automated Circuit Discovery for
    Mechanistic Interpretability' (https://arxiv.org/abs/2304.14997).

    The algorithm does not assign scores to each edge, instead it finds the edge to be
    pruned given a certain value of tao. So we run the algorithm for several values of
    tao and give equal scores to all edges that are pruned for a given tao (with a
    greater score for lower values of tao). Then we pass test_edge_counts to run_pruned
    such that all edges with the same score are pruned together."""
    edges: OrderedSet[Edge] = graph_edges(model, factorized)
    edges = OrderedSet(reversed(list(edges)))
    exp_type = ExperimentType(ActType.CLEAN, ActType.CORRUPT)
    tao_values = t.arange(tao_range[0], tao_range[1] + tao_step, tao_step)
    prune_scores = dict([(edge, float("inf")) for edge in edges])
    for tao in (pbar_tao := tqdm(tao_values)):
        tao = tao.item()
        pbar_tao.set_description_str("ACDC \u03C4={:.7f}".format(tao))
        edges_to_prune: Dict[Edge, float] = {}
        prev_kl_div = 0.0
        for edge in (pbar_edge := tqdm(edges)):
            desc = f"ACDC Included Edges={len(edges_to_prune)}, Current Edge='{edge}'"
            pbar_edge.set_description_str(desc)
            edges_to_prune[edge] = 1.0
            edge_count = len(edges_to_prune)
            pruned_outs = run_pruned(
                model, factorized, train_data, exp_type, [edge_count], edges_to_prune
            )
            kl_divs_clean, _ = measure_kl_div(model, train_data, pruned_outs)
            kl_div_clean = kl_divs_clean[edge_count]
            if kl_div_clean - prev_kl_div < tao:
                prev_kl_div = kl_div_clean
                prune_scores[edge] = min(tao, prune_scores[edge])
            else:
                del edges_to_prune[edge]
    return prune_scores


def acdc_edge_counts(
    model: t.nn.Module, factorized: bool, prune_scores: Dict[Edge, float]
) -> List[int]:
    # Group prune_scores by score
    prune_scores_count: Dict[float, int] = defaultdict(int)
    for _, score in prune_scores.items():
        prune_scores_count[score] += 1
    # Sort prune_scores by score
    prune_scores_count = dict(
        sorted(prune_scores_count.items(), key=lambda item: item[0])
    )
    # Create edge_counts
    edge_counts = []
    for _, count in prune_scores_count.items():
        prev_count = edge_counts[-1] if edge_counts else 0
        edge_counts += [prev_count + count]
    return edge_counts_util(model, factorized, edge_counts)
