from functools import partial

from auto_circuit.prune_algos.ACDC import acdc_prune_scores
from auto_circuit.prune_algos.activation_magnitude import (
    activation_magnitude_prune_scores,
)
from auto_circuit.prune_algos.edge_attribution_patching import (
    edge_attribution_patching_prune_scores,
)
from auto_circuit.prune_algos.ground_truth import ground_truth_prune_scores
from auto_circuit.prune_algos.integrated_edge_gradients import (
    integrated_edge_gradients_prune_scores,
)
from auto_circuit.prune_algos.random_edges import random_prune_scores
from auto_circuit.prune_algos.simple_gradient import simple_gradient_prune_scores
from auto_circuit.prune_algos.subnetwork_probing import subnetwork_probing_prune_scores
from auto_circuit.types import PruneAlgo

GROUND_TRUTH_PRUNE_ALGO = PruneAlgo(
    "Ground Truth", ground_truth_prune_scores, short_name="GT"
)
# f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
#     parameter_integrated_grads_prune_scores,
#     baseline_weights=pig_baseline,
#     samples=pig_samples,
# ),
ACT_MAG_PRUNE_ALGO = PruneAlgo(
    "Activation Magnitude", activation_magnitude_prune_scores
)
RANDOM_PRUNE_ALGO = PruneAlgo("Random", random_prune_scores)
EDGE_ATTR_PATCH_PRUNE_ALGO = PruneAlgo(
    "Edge Attribution Patching", edge_attribution_patching_prune_scores
)
ACDC_PRUNE_ALGO = PruneAlgo(
    "ACDC",
    partial(
        acdc_prune_scores,
        # tao_exps=list(range(-6, 1)),
        tao_exps=[-5],
        tao_bases=[1],
    ),
)
INTEGRATED_EDGE_GRADS_PRUNE_ALGO = PruneAlgo(
    "Integrated Edge Gradients (answer grad)",
    partial(
        integrated_edge_gradients_prune_scores,
        samples=50,
    ),
)
INTEGRATED_EDGE_GRADS_LOGIT_DIFF_PRUNE_ALGO = PruneAlgo(
    "Integrated Edge Gradients",
    partial(
        integrated_edge_gradients_prune_scores,
        samples=50,
        answer_diff=True,
    ),
)
PROB_GRAD_PRUNE_ALGO = PruneAlgo(
    "Prob Gradient", partial(simple_gradient_prune_scores, grad_function="prob")
)
LOGIT_EXP_GRAD_PRUNE_ALGO = PruneAlgo(
    "Exp Logit Gradient",
    partial(simple_gradient_prune_scores, grad_function="logit_exp"),
)
LOGPROB_GRAD_PRUNE_ALGO = PruneAlgo(
    "Logprob Gradient",
    partial(simple_gradient_prune_scores, grad_function="logprob", answer_diff=False),
)
LOGPROB_DIFF_GRAD_PRUNE_ALGO = PruneAlgo(
    "Edge Attribution Patching",
    partial(
        simple_gradient_prune_scores,
        grad_function="logprob",
        answer_diff=True,
        mask_val=0.0,
    ),
)  # USE THIS
SUBNETWORK_EDGE_PROBING_PRUNE_ALGO = PruneAlgo(
    "Subnetwork Edge Probing",
    partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=1000,
        regularize_lambda=0.5,
        mask_fn="hard_concrete",
        dropout_p=0.0,
        show_train_graph=True,
    ),
)
CIRCUIT_PROBING_PRUNE_ALGO = PruneAlgo(
    "Circuit Probing",
    partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=500,
        regularize_lambda=0.1,
        mask_fn="hard_concrete",
        dropout_p=0.0,
        show_train_graph=True,
        regularize_to_true_circuit_size=True,
    ),
    short_name="CP",
)

SUBNETWORK_TREE_PROBING_PRUNE_ALGO = PruneAlgo(
    "250*4 TREE Subnetwork Edge Probing",
    partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=250,
        regularize_lambda=0.5,
        mask_fn="hard_concrete",
        dropout_p=0.0,
        show_train_graph=True,
        tree_optimisation=True,
    ),
)
CIRCUIT_TREE_PROBING_PRUNE_ALGO = PruneAlgo(
    "250*4 TREE Circuit Probing",
    partial(
        subnetwork_probing_prune_scores,
        learning_rate=0.1,
        epochs=250,
        regularize_lambda=0.1,
        mask_fn="hard_concrete",
        dropout_p=0.0,
        show_train_graph=True,
        regularize_to_true_circuit_size=True,
        tree_optimisation=True,
    ),
    short_name="250*4 TREE CP",
)
