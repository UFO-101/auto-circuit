from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Set

from auto_circuit.data import PromptDataLoader
from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.prune_algos.activation_magnitude import (
    activation_magnitude_prune_scores,
)
from auto_circuit.prune_algos.circuit_probing import circuit_probing_prune_scores
from auto_circuit.prune_algos.edge_attribution_patching import (
    edge_attribution_patching_prune_scores,
)
from auto_circuit.prune_algos.ground_truth import ground_truth_prune_scores
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.prune_algos.random_edges import random_prune_scores
from auto_circuit.prune_algos.subnetwork_probing import subnetwork_probing_prune_scores
from auto_circuit.tasks import Task
from auto_circuit.types import (
    AlgoKey,
    AlgoPruneScores,
    Edge,
    PruneScores,
    TaskPruneScores,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.patchable_model import PatchableModel


@dataclass(frozen=True)
class PruneAlgo:
    """
    An algorithm that finds the importance of each edge in a model for a given task.

    Args:
        key: A unique identifier for the algorithm.
        name: The name of the algorithm.
        func: The function that computes the importance of each edge.
        _short_name: A short name for the algorithm. If not provided, `name` is used.
    """

    key: AlgoKey
    name: str
    func: Callable[[PatchableModel, PromptDataLoader, Optional[Set[Edge]]], PruneScores]
    _short_name: Optional[str] = None

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PruneAlgo):
            return False
        return self.key == __value.key

    @property
    def short_name(self) -> str:
        return self._short_name if self._short_name is not None else self.name


def run_prune_algos(tasks: List[Task], prune_algos: List[PruneAlgo]) -> TaskPruneScores:
    """
    Run a list of pruning algorithms on a list of tasks.

    Args:
        tasks: The tasks to run the algorithms on.
        prune_algos: The algorithms to run on the tasks.

    Returns:
        A nested dictionary of the prune scores for each task and algorithm.
    """
    task_prune_scores: TaskPruneScores = {}
    for task in (experiment_pbar := tqdm(tasks)):
        experiment_pbar.set_description_str(f"Task: {task.name}")
        prune_scores_dict: AlgoPruneScores = {}
        for prune_algo in (prune_score_pbar := tqdm(prune_algos)):
            prune_score_pbar.set_description_str(f"Prune scores: {prune_algo.name}")
            ps = prune_algo.func(task.model, task.train_loader, task.true_edges)
            prune_scores_dict[prune_algo.key] = ps
        task_prune_scores[task.key] = prune_scores_dict
    return task_prune_scores


GROUND_TRUTH_PRUNE_ALGO = PruneAlgo(
    key="Official Circuit",
    name="Ground Truth",
    func=ground_truth_prune_scores,
    _short_name="GT",
)
# f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
#     parameter_integrated_grads_prune_scores,
#     baseline_weights=pig_baseline,
#     samples=pig_samples,
# ),
ACT_MAG_PRUNE_ALGO = PruneAlgo(
    key="Act Mag",
    name="Activation Magnitude",
    _short_name="Act Mag",
    func=activation_magnitude_prune_scores,
)
RANDOM_PRUNE_ALGO = PruneAlgo(key="Random", name="Random", func=random_prune_scores)
EDGE_ATTR_PATCH_PRUNE_ALGO = PruneAlgo(
    key="Edge Attribution Patching",
    name="Edge Attribution Patching",
    _short_name="EAP",
    func=edge_attribution_patching_prune_scores,
)
ACDC_PRUNE_ALGO = PruneAlgo(
    key="ACDC",
    name="ACDC",
    func=partial(
        acdc_prune_scores,
        tao_exps=list(range(-6, -2)),
        # tao_exps=[-5],
        tao_bases=[1],
    ),
)
MSE_ACDC_PRUNE_ALGO = PruneAlgo(
    key="MSE ACDC",
    name="MSE ACDC",
    _short_name="ACDC (MSE)",
    func=partial(
        acdc_prune_scores,
        tao_bases=[1],
        tao_exps=list(range(-10, 0)),
        faithfulness_target="mse",
    ),
)
INTEGRATED_EDGE_GRADS_PRUNE_ALGO = PruneAlgo(
    key="Integrated Edge Gradients (Logit Grad)",
    name="Integrated Edge Gradients (Logit Grad)",
    _short_name="IEG (Answer Logit)",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logit",
        answer_function="avg_val",
        integrated_grad_samples=50,
    ),
)
INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO = PruneAlgo(
    key="Integrated Edge Gradients (Logit Diff)",
    name="Integrated Edge Gradients",
    _short_name="IEG",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logit",
        answer_function="avg_diff",
        integrated_grad_samples=1000,
    ),
)
PROB_GRAD_PRUNE_ALGO = PruneAlgo(
    key="Edge Answer Prob Gradient At Clean",
    name="Prob Gradient",
    _short_name="EAP (Prob)",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="prob",
        answer_function="avg_val",
        mask_val=0.0,
    ),
)
LOGIT_EXP_GRAD_PRUNE_ALGO = PruneAlgo(
    key="Edge Answer Logit Exp Gradient At Clean",
    name="Exp Logit Gradient",
    _short_name="EAP (Answer Logit)",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logit_exp",
        answer_function="avg_val",
        mask_val=0.0,
    ),
)
LOGPROB_GRAD_PRUNE_ALGO = PruneAlgo(
    key="Edge Answer Log Prob Gradient At Clean",
    name="Logprob Gradient",
    _short_name="EAP (Answer Logprob)",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logprob",
        answer_function="avg_val",
        mask_val=0.0,
    ),
)
LOGPROB_DIFF_GRAD_PRUNE_ALGO = PruneAlgo(
    key="Edge Answer Log Prob Diff Gradient At Clean",
    name="Logprob Diff Gradient",
    _short_name="EAP (Logprob Diff)",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logprob",
        answer_function="avg_diff",
        mask_val=0.0,
    ),
)
LOGIT_DIFF_GRAD_PRUNE_ALGO = PruneAlgo(
    key="Edge Answer Logit Diff Gradient At Clean",
    name="Edge Attribution Patching",
    _short_name="EAP",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logit",
        answer_function="avg_diff",
        mask_val=0.0,
    ),
)
LOGIT_MSE_GRAD_PRUNE_ALGO = PruneAlgo(
    key="Edge Answer Logit MSE Gradient At Clean",
    name="Edge Attribution Patching (MSE)",
    _short_name="EAP (MSE)",
    func=partial(
        mask_gradient_prune_scores,
        grad_function="logit",
        answer_function="mse",
        mask_val=0.0,
    ),
)
SUBNETWORK_EDGE_PROBING_PRUNE_ALGO = PruneAlgo(
    key="Subnetwork Edge Probing",
    name="Subnetwork Edge Probing",
    _short_name="SEP",
    func=partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=200,
        regularize_lambda=0.5,
        mask_fn="hard_concrete",
        show_train_graph=True,
    ),
)
CIRCUIT_PROBING_PRUNE_ALGO = PruneAlgo(
    key="Circuit Probing",
    name="Circuit Probing",
    _short_name="CP",
    func=partial(
        circuit_probing_prune_scores,
        learning_rate=0.1,
        epochs=2000,
        regularize_lambda=0.1,
        mask_fn="hard_concrete",
        show_train_graph=True,
        circuit_sizes=["true_size", 1000],
    ),
)
SUBNETWORK_TREE_PROBING_PRUNE_ALGO = PruneAlgo(
    key="Subnetwork Tree Probing",
    name="Subnetwork Tree Probing",
    _short_name="STP",
    func=partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=2000,
        regularize_lambda=0.5,
        mask_fn="hard_concrete",
        show_train_graph=False,
        tree_optimisation=True,
    ),
)
CIRCUIT_TREE_PROBING_PRUNE_ALGO = PruneAlgo(
    key="Tree Probing",
    name="Tree Probing",
    _short_name="TP",
    func=partial(
        circuit_probing_prune_scores,
        learning_rate=0.1,
        epochs=200,
        regularize_lambda=0.1,
        mask_fn="hard_concrete",
        show_train_graph=False,
        circuit_sizes=[100, "true_size", 1000, 10000],
        tree_optimisation=True,
    ),
)
MSE_SUBNETWORK_TREE_PROBING_PRUNE_ALGO = PruneAlgo(
    key="MSE Subnetwork Tree Probing",
    name="MSE Subnetwork Tree Probing",
    _short_name="STP (MSE)",
    func=partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=200,
        regularize_lambda=0.5,
        mask_fn="hard_concrete",
        show_train_graph=True,
        faithfulness_target="mse",
        tree_optimisation=True,
    ),
)
MSE_CIRCUIT_TREE_PROBING_PRUNE_ALGO = PruneAlgo(
    key="MSE Tree Probing",
    name="MSE Tree Probing",
    _short_name="TP (MSE)",
    func=partial(
        circuit_probing_prune_scores,
        learning_rate=0.1,
        epochs=200,
        regularize_lambda=0.1,
        mask_fn="hard_concrete",
        show_train_graph=True,
        circuit_sizes=["true_size"],
        tree_optimisation=True,
        faithfulness_target="mse",
    ),
)
OPPOSITE_TREE_PROBING_PRUNE_ALGO = PruneAlgo(
    key="Opposite Tree Probing",
    name="Opposite Tree Probing",
    _short_name="OTP",
    func=partial(
        circuit_probing_prune_scores,
        learning_rate=0.1,
        epochs=100,
        regularize_lambda=0.1,
        mask_fn="hard_concrete",
        show_train_graph=True,
        circuit_sizes=["true_size", 1000, 10000],
        tree_optimisation=True,
        faithfulness_target="wrong_answer",
    ),
)

PRUNE_ALGOS: List[PruneAlgo] = [
    GROUND_TRUTH_PRUNE_ALGO,
    ACT_MAG_PRUNE_ALGO,
    RANDOM_PRUNE_ALGO,
    EDGE_ATTR_PATCH_PRUNE_ALGO,
    ACDC_PRUNE_ALGO,
    MSE_ACDC_PRUNE_ALGO,
    INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO,
    LOGPROB_GRAD_PRUNE_ALGO,
    LOGPROB_DIFF_GRAD_PRUNE_ALGO,
    LOGIT_DIFF_GRAD_PRUNE_ALGO,
    LOGIT_MSE_GRAD_PRUNE_ALGO,
    SUBNETWORK_EDGE_PROBING_PRUNE_ALGO,
    CIRCUIT_PROBING_PRUNE_ALGO,
    SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    CIRCUIT_TREE_PROBING_PRUNE_ALGO,
    MSE_SUBNETWORK_TREE_PROBING_PRUNE_ALGO,
    MSE_CIRCUIT_TREE_PROBING_PRUNE_ALGO,
    OPPOSITE_TREE_PROBING_PRUNE_ALGO,
]
PRUNE_ALGO_DICT: Dict[AlgoKey, PruneAlgo] = {algo.key: algo for algo in PRUNE_ALGOS}
