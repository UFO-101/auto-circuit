from collections import defaultdict
from typing import Dict, List

import torch as t
from plotly import graph_objects as go
from plotly import subplots

from auto_circuit.prune_algos.prune_algos import PRUNE_ALGO_DICT
from auto_circuit.tasks import TASK_DICT
from auto_circuit.types import (
    AlgoKey,
    AlgoPruneScores,
    TaskKey,
    TaskPruneScores,
)
from auto_circuit.utils.tensor_ops import desc_prune_scores, prune_scores_threshold


def prune_score_similarities(
    algo_prune_scores: AlgoPruneScores, edge_counts: List[int]
) -> Dict[int, Dict[AlgoKey, Dict[AlgoKey, float]]]:
    """
    Measure the similarity between the prune scores of different algorithms. Similarity
    is measured as the proportion of the top N prune scores that are common between two
    algorithms.

    Args:
        algo_prune_scores: Prune scores for each algorithm.
        edge_counts: The number of edges in the circuit to consider.

    Returns:
        A dictionary mapping from the number of edges considered to the overlap between
            [`PruneScores`][auto_circuit.types.PruneScores] for each pair of algorithms.
    """
    desc_ps: Dict[AlgoKey, t.Tensor] = {}
    for algo_key, prune_scores in algo_prune_scores.items():
        desc_ps[algo_key] = desc_prune_scores(prune_scores)

    similarity_scores: Dict[int, Dict[AlgoKey, Dict[AlgoKey, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for algo_key_1, desc_ps_1 in desc_ps.items():
        for algo_key_2, desc_ps_2 in desc_ps.items():
            if algo_key_2 in similarity_scores[edge_counts[0]]:
                continue
            for edge_count in edge_counts:
                threshold_1 = prune_scores_threshold(desc_ps_1, edge_count)
                threshold_2 = prune_scores_threshold(desc_ps_2, edge_count)
                # Find the edges common to the top N prune scores for each algorithm
                real_edge_count_1, real_edge_count_2 = 0, 0
                common_edge_count = 0
                for mod, ps_1 in algo_prune_scores[algo_key_1].items():
                    ps_2 = algo_prune_scores[algo_key_2][mod]
                    circuit_1 = ps_1.abs() >= threshold_1
                    circuit_2 = ps_2.abs() >= threshold_2
                    real_edge_count_1 += circuit_1.sum().item()
                    real_edge_count_2 += circuit_2.sum().item()
                    common_edge_count += (circuit_1 & circuit_2).sum().item()

                if real_edge_count_1 != edge_count or real_edge_count_2 != edge_count:
                    similarity_scores[edge_count][algo_key_1][algo_key_2] = float("nan")
                else:
                    similarity_scores[edge_count][algo_key_1][algo_key_2] = (
                        common_edge_count / edge_count
                    )

    return similarity_scores


def task_prune_scores_similarities(
    task_prune_scores: TaskPruneScores,
    edge_counts: List[int],
    true_edge_counts: bool = False,
) -> Dict[TaskKey, Dict[int, Dict[AlgoKey, Dict[AlgoKey, float]]]]:
    """
    Wrapper around
    [`prune_score_similarities`][auto_circuit.metrics.prune_scores_similarity.prune_score_similarities]
    for a set of tasks.

    Args:
        true_edge_counts: Whether to include the official circuit edge count in addition
            to `edge_counts`.
    """
    task_similarity: Dict[TaskKey, Dict[int, Dict[AlgoKey, Dict[AlgoKey, float]]]] = {}

    for task_key, algo_prune_scores in task_prune_scores.items():
        task = TASK_DICT[task_key]
        true_edge_count = []
        if true_edge_counts:
            assert task.true_edges is not None
            true_edge_count = [len(task.true_edges)]
        task_similarity[task_key] = prune_score_similarities(
            algo_prune_scores, true_edge_count + edge_counts
        )

    return task_similarity


def prune_score_similarities_plotly(
    task_prune_scores: TaskPruneScores,
    edge_counts: List[int],
    ground_truths: bool = False,
) -> go.Figure:
    """
    Create a Plotly heatmap figure showing the similarity between the prune scores of
    different algorithms.

    See
    [`prune_score_similarities`][auto_circuit.metrics.prune_scores_similarity.prune_score_similarities]

    Args:
        task_prune_scores: Prune scores for each algorithm for each task.
        edge_counts: The number of edges in the circuit to consider.
        ground_truths: Whether to include the official circuit edge counts in addition
            to `edge_counts`.

    Returns:
        A Plotly figure.
    """
    sims = task_prune_scores_similarities(task_prune_scores, edge_counts, ground_truths)

    row_count = len(sims)
    col_count = len(edge_counts) + (1 if ground_truths else 0)
    algo_count = 0
    fig = subplots.make_subplots(
        rows=row_count,
        cols=col_count,
        shared_xaxes=True,
        shared_yaxes=True,
        row_titles=[TASK_DICT[task_key].name for task_key in sims.keys()],
        column_titles=(["|Ground Truth| Edges"] if ground_truths else [])
        + [f"{edge_count} Edges" for edge_count in edge_counts],
    )
    for task_idx, edge_count_sims in enumerate(sims.values()):
        for count_idx, algo_sims in enumerate(edge_count_sims.values()):
            algo_count = len(algo_sims)
            x_strs = [PRUNE_ALGO_DICT[a].short_name for a in reversed(algo_sims.keys())]
            y_strs = [PRUNE_ALGO_DICT[algo].short_name for algo in algo_sims.keys()]
            heatmap = []
            for similarity_dict in algo_sims.values():
                row = [sim_score for sim_score in similarity_dict.values()]
                heatmap.append(list(reversed(row)))
            fig.add_trace(
                go.Heatmap(
                    x=x_strs,
                    y=y_strs,
                    z=heatmap,
                    colorscale="Viridis",
                    showscale=False,
                    text=heatmap,
                    texttemplate="%{text:.0%}",
                    textfont={"size": 19},
                ),
                row=task_idx + 1,
                col=count_idx + 1,
            )
    # fig.update_layout(yaxis_scaleanchor="x")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        width=col_count * 70 * algo_count + 100,
        height=row_count * 50 * algo_count + 100,
    )
    return fig
