#%%
import pickle
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import plotly.graph_objects as go
import torch as t
import transformer_lens as tl

import auto_circuit.data
from auto_circuit.metrics.answer_diff import measure_answer_diff
from auto_circuit.metrics.answer_value import measure_answer_val
from auto_circuit.metrics.kl_div import measure_kl_div
from auto_circuit.metrics.official_circuits.docstring_official import (
    docstring_true_edges,
)
from auto_circuit.metrics.ROC import measure_roc
from auto_circuit.prune import run_pruned
from auto_circuit.prune_functions.random_edges import random_prune_scores
from auto_circuit.prune_functions.simple_gradient import simple_gradient_prune_scores
from auto_circuit.types import (
    Y_MIN,
    AlgoPruneScores,
    Edge,
    ExperimentPruneScores,
    Measurements,
    MetricMeasurements,
    PatchType,
    Task,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import edge_counts_util, prepare_model
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import average_auc_plot, edge_patching_plot, roc_plot


def create_transformer_lens_experiment(
    name: str,
    model_name: str,
    dataset: str,
    true_edge_func: Callable[..., Set[Edge]],
    token_circuit: bool = False,
) -> Task:
    device = "cuda" if t.cuda.is_available() else "cpu"

    model = tl.HookedTransformer.from_pretrained(model_name, device=device)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    assert model.tokenizer is not None
    model.eval()

    train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
        tokenizer=model.tokenizer,
        path=repo_path_to_abs_path(f"datasets/{dataset}.json"),
        device=device,
        prepend_bos=True,
        batch_size=32,
        train_test_split=[0.5, 0.5],
        length_limit=64,
        return_seq_length=token_circuit,
        pad=True,
    )
    seq_len = train_loader.seq_len
    prepare_model(
        model, factorized=True, slice_output=True, seq_len=seq_len, device=device
    )

    return Task(
        name,
        model,
        train_loader,
        test_loader,
        true_edge_func,
        token_circuit,
    )


def transformer_experiments(token_circuits: bool) -> List[Task]:
    experiments = [
        # create_transformer_lens_experiment(
        #     name="Indirect Object Identification",
        #     model_name="gpt2-small",
        #    dataset="ioi_single_template_prompts" if token_circuits else "ioi_prompts",
        #     true_edge_func=ioi_true_edges,
        #     token_circuit=token_circuits,
        # ),
        create_transformer_lens_experiment(
            name="Docstring",
            model_name="attn-only-4l",
            dataset="docstring_prompts",
            true_edge_func=docstring_true_edges,
            token_circuit=token_circuits,
        ),
    ]
    # if not token_circuits:
    #     experiments.append(create_transformer_lens_experiment(
    #         name="Greaterthan",
    #         model_name="gpt2-small",
    #         dataset="greater_than_gpt2-small_prompts",
    #         true_edge_func=greaterthan_true_edges,
    #         token_circuit=token_circuits,
    #     ))
    return experiments


PRUNE_FUNCS: Dict[str, Callable] = {
    # f"PIG ({pig_baseline.name.lower()} Base, {pig_samples} iter)": partial(
    #     parameter_integrated_grads_prune_scores,
    #     baseline_weights=pig_baseline,
    #     samples=pig_samples,
    # ),
    # "Act Mag": activation_magnitude_prune_scores,
    "Random": random_prune_scores,
    # "Edge Attribution Patching": edge_attribution_patching_prune_scores,
    # "ACDC": partial(
    #     acdc_prune_scores,
    #     # tao_exps=list(range(-6, 1)),
    #     tao_exps=[-5],
    #     tao_bases=[1],
    # ),
    # "Integrated edge gradients": partial(
    #     integrated_edge_gradients_prune_scores,
    #     samples=50,
    # ),
    # "Integrated Edge Gradients": partial(
    #     integrated_edge_gradients_prune_scores,
    #     samples=50,
    #     answer_diff=True,
    # ),
    # "Prob Gradient": partial(simple_gradient_prune_scores, grad_function="prob"),
    # "Exp Logit Gradient": partial(
    #     simple_gradient_prune_scores, grad_function="logit_exp"
    # ),
    "Logprob Gradient": partial(
        simple_gradient_prune_scores, grad_function="logprob", answer_diff=True
    ),  # USE THIS
    # "Subnetwork Edge Probing": partial(
    #     subnetwork_probing_prune_scores,
    #     learning_rate=0.1,
    #     epochs=500,
    #     regularize_lambda=0.5,
    #     mask_fn="hard_concrete",
    #     dropout_p=0.0,
    #     show_train_graph=True,
    # ),
}


def run_prune_funcs(experiments: List[Task]) -> ExperimentPruneScores:
    experiment_prune_scores: ExperimentPruneScores = {}
    for experiment in (experiment_pbar := tqdm(experiments)):
        experiment_pbar.set_description_str(f"Task: {experiment.name}")
        prune_scores_dict: AlgoPruneScores = {}
        prune_scores_dict["Ground Truth"] = dict(
            [(edge, 1.0) for edge in experiment.true_edges]
        )
        for name, prune_func in (prune_score_pbar := tqdm(PRUNE_FUNCS.items())):
            prune_score_pbar.set_description_str(f"Prune scores: {name}")
            new_prune_scores = prune_func(experiment.model, experiment.train_loader)
            if name in prune_scores_dict:
                prune_scores_dict[name].update(new_prune_scores)
            else:
                prune_scores_dict[name] = new_prune_scores
        experiment_prune_scores[experiment] = prune_scores_dict
    return experiment_prune_scores


def measure_experiment_metrics(
    all_prune_scores: ExperimentPruneScores,
) -> MetricMeasurements:
    measurements: MetricMeasurements = defaultdict(lambda: defaultdict(dict))
    for experiment, algo_prune_scores in (exp_pbar := tqdm(all_prune_scores.items())):
        exp_pbar.set_description_str(f"Task: {experiment.name}")
        model = experiment.model
        edges: Set[Edge] = model.edges  # type: ignore
        test_loader = experiment.test_loader
        for func_name, prune_scores in (func_pbar := tqdm(algo_prune_scores.items())):
            func_pbar.set_description_str(f"Pruning with {func_name}")

            pruned_outs = run_pruned(
                model=model,
                dataloader=test_loader,
                test_edge_counts=edge_counts_util(edges, prune_scores=prune_scores),
                prune_scores=prune_scores,
                patch_type=PatchType.TREE_PATCH,
                render_graph=False,
                render_prune_scores=True,
                render_top_n=30,
                render_file_path="figures-6/docstring-viz.pdf",
            )
            kl_points: Measurements = measure_kl_div(model, test_loader, pruned_outs)
            answer_prob_points: Measurements = measure_answer_val(
                model, test_loader, pruned_outs, prob_func="softmax"
            )
            logit_diff_points: Measurements = measure_answer_diff(
                model, test_loader, pruned_outs, "logit"
            )
            del pruned_outs
            measurements["kl_div"][experiment.name][func_name] = kl_points
            measurements["answer_prob"][experiment.name][func_name] = answer_prob_points
            measurements["logit_diff"][experiment.name][func_name] = logit_diff_points

            roc_points: Measurements = measure_roc(
                model=model,
                prune_scores=prune_scores,
                correct_edges=experiment.true_edges,
            )
            measurements["roc"][experiment.name][func_name] = roc_points

            t.cuda.empty_cache()
    return measurements


def measurement_figs(
    measurements: MetricMeasurements, token_circuit: bool
) -> Tuple[go.Figure, ...]:
    figs = []
    for metric, task_measurements in measurements.items():
        data, y_max = [], 0.0
        for experiment_name, algo_measurements in task_measurements.items():
            for algo, points in algo_measurements.items():
                if len(points) > 1:
                    for x, y in points:
                        data.append(
                            {
                                "Task": experiment_name,
                                "Algorithm": algo,
                                "X": x,
                                "Y": max(y, Y_MIN),
                            }
                        )
                        y_max = max(y_max, y)
        if metric == "kl_div":
            figs.append(
                edge_patching_plot(
                    data, task_measurements, "KL Divergence", True, token_circuit, y_max
                )
            )
        elif metric == "answer_prob":
            figs.append(
                edge_patching_plot(
                    data, task_measurements, "Answer Probability", False, token_circuit
                )
            )
        elif metric == "logit_diff":
            figs.append(
                edge_patching_plot(
                    data, task_measurements, "Logit Difference", False, token_circuit
                )
            )
        else:
            assert metric == "roc"
            figs.append(roc_plot(data, token_circuit))
        metric_names = {
            "kl_div": ("KL Divergence", True, True),
            "answer_prob": ("Answer Probability", True, False),
            "logit_diff": ("Logit Difference", False, False),
            "roc": ("ROC", False, False),
        }
        metric_name, log_xy, inverse = metric_names[metric]
        figs.append(average_auc_plot(task_measurements, metric_name, log_xy, inverse))
    return tuple(figs)


TOKEN_CIRCUITS = True
experiment_prune_scores = run_prune_funcs(transformer_experiments(TOKEN_CIRCUITS))
measurements = measure_experiment_metrics(experiment_prune_scores)


# Cache measurements with current date and time
save = False
load = False
if save:
    now = datetime.now()
    # dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    # repo_path = f".measurement_cache/seq_experiment_measurements-{dt_string}.pkl"
    repo_path = ".measurement_cache/token_pos_tree_patch_2.pkl"
    with open(repo_path_to_abs_path(repo_path), "wb") as f:
        pickle.dump(dict(measurements), f)
if load:
    # cache_path = "experiment_measurements-26-11-2023_16-36-05.pkl"
    # cache_path = "seq_experiment_measurements-28-11-2023_15-52-47.pkl"
    # cache_path = "token_pos_exp_1.pkl"
    cache_path = "token_pos_tree_patch_2.pkl"
    with open(repo_path_to_abs_path(".measurement_cache/" + cache_path), "rb") as f:
        loaded_measurements = pickle.load(f)
    # merge with existing measurements
    for metric, loaded_task_measurements in loaded_measurements.items():
        for task_name, loaded_algo_measurements in loaded_task_measurements.items():
            measurements[metric][task_name].update(loaded_algo_measurements)

# experiment_steps: Dict[str, Callable] = {
#     "Calculate prune scores": run_prune_funcs,
#     "Measure experiment metric": measure_experiment_metrics,
#     "Draw figures": measurement_figs
# }
figs = measurement_figs(measurements, TOKEN_CIRCUITS)
for i, fig in enumerate(figs):
    fig.show()
    folder: Path = repo_path_to_abs_path("figures-5")
    # Save figure as pdf in figures folder
    # fig.write_image(str(folder / f"{i}.pdf"))

#%%
