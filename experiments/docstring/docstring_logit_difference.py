#%%
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import torch as t
from plotly import subplots
from torch.nn.functional import softmax

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import (
    load_tl_model,
)
from auto_circuit.metrics.official_circuits.circuits.docstring_official import (
    docstring_node_based_official_edges,
    docstring_true_edges,
)
from auto_circuit.metrics.prune_metrics.answer_value import measure_answer_val
from auto_circuit.prune import run_circuits
from auto_circuit.types import (
    COLOR_PALETTE,
    AblationType,
    CircuitOutputs,
    Measurements,
    PatchType,
)
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    edge_counts_util,
    patchable_model,
)
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.tensor_ops import indices_vals

#%%

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
model = load_tl_model("attn-only-4l", device)

path = repo_path_to_abs_path("datasets/docstring_prompts.json")
_, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=128,
    train_test_size=(0, 128),
    shuffle=False,
    return_seq_length=True,
    tail_divergence=True,
)


#%%

# Mapping from [token_positions, edge_circuit] to the proportion of correct answers
ablation_types: List[AblationType] = [
    AblationType.RESAMPLE,
    AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
]
points: Dict[AblationType, Dict[Tuple[bool, bool], t.Tensor]] = defaultdict(dict)
model_probs = None

for ablation_type in tqdm(ablation_types):
    for tok_pos_circuit in tqdm([False, True]):
        for edge_circuit in tqdm([False, True]):
            if edge_circuit:
                circ_edge_func = docstring_true_edges
            else:
                circ_edge_func = docstring_node_based_official_edges

            patch_model = deepcopy(model)
            patch_model = patchable_model(
                model=patch_model,
                factorized=edge_circuit,
                slice_output="last_seq",
                seq_len=test_loader.seq_len if tok_pos_circuit else None,
                separate_qkv=True,
                kv_caches=(test_loader.kv_cache,),
                device=device,
            )

            docstring_edges = circ_edge_func(
                patch_model,
                word_idxs=test_loader.word_idxs,
                token_positions=tok_pos_circuit,
                seq_start_idx=test_loader.diverge_idx,
            )
            prune_scores = patch_model.circuit_prune_scores(docstring_edges)

            # draw_seq_graph(
            #     model=patch_model,
            #     prune_scores=prune_scores,
            #     seq_labels=test_loader.seq_labels
            # )

            circuit_outs: CircuitOutputs = run_circuits(
                model=patch_model,
                dataloader=test_loader,
                test_edge_counts=edge_counts_util(
                    patch_model.edges, prune_scores=prune_scores, all_edges=True
                ),
                prune_scores=prune_scores,
                patch_type=PatchType.TREE_PATCH,
                ablation_type=ablation_type,
            )
            measurements: Measurements = measure_answer_val(
                model=patch_model,
                test_loader=test_loader,
                pruned_outs=circuit_outs,
                prob_func="softmax",
            )

            circ_probs = []
            for batch in test_loader:
                circ_batch_logits = circuit_outs[len(docstring_edges)][batch.key]
                batch_probs = indices_vals(
                    softmax(circ_batch_logits, dim=-1), batch.answers
                )
                circ_probs.append(batch_probs)
            circ_probs = t.cat(circ_probs, dim=0)
            points[ablation_type][(tok_pos_circuit, edge_circuit)] = circ_probs
            circ_probs_mean = circ_probs.mean().item()
            circ_probs_std = circ_probs.std().item()

            assert len(measurements) == 2
            n_edge, avg_correct = measurements[0]
            assert circ_probs_mean == avg_correct

            # Also get the score for the full model
            model_probs = []
            for batch in test_loader:
                with t.no_grad():
                    model_batch_logits = patch_model(batch.clean)[patch_model.out_slice]
                model_batch_probs = indices_vals(
                    softmax(model_batch_logits, dim=-1), batch.answers
                )
                model_probs.append(model_batch_probs)
            model_probs = t.cat(model_probs, dim=0)
            model_probs_mean = model_probs.mean().item()
            model_probs_std = model_probs.std().item()

            all_edges, full_model_avg_correct = measurements[1]
            print("all_edges", all_edges, "patch_model.n_edges", patch_model.n_edges)
            assert all_edges == patch_model.n_edges
            print(f"Model: {model_probs_mean}, Circuit: {circ_probs_mean}")
            print(
                "model_probs_mean",
                model_probs_mean,
                "full_model_avg_correct",
                full_model_avg_correct,
            )
            # assert model_probs_mean == full_model_avg_correct

            del patch_model

#%%
def bar_name(tok_pos_circuit: bool, edge_circuit: bool) -> str:
    name = ""
    name += "Edges " if edge_circuit else "Nodes "
    name += "(tokens)" if tok_pos_circuit else ""
    return name


fig = go.Figure()
fig = subplots.make_subplots(
    rows=1,
    cols=len(ablation_types),
    column_titles=[str(a) for a in ablation_types],
    shared_yaxes=True,
)
for i, ablation_type in enumerate(ablation_types):
    for j, (tok_pos_circuit, edge_circuit) in enumerate(points[ablation_type].keys()):
        circ_points = points[ablation_type][(tok_pos_circuit, edge_circuit)]
        fig.add_trace(
            go.Box(
                x=[bar_name(tok_pos_circuit, edge_circuit)] * circ_points.shape[0],
                y=circ_points.squeeze().cpu().tolist(),
                name=f"{tok_pos_circuit} {edge_circuit}",
                boxpoints="all",
                marker=dict(
                    color=COLOR_PALETTE[j],
                    opacity=0.25,
                ),
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )
    assert model_probs is not None
    fig.add_trace(
        go.Box(
            x=["Full Model"] * model_probs.shape[0],
            y=model_probs.squeeze().cpu().tolist(),
            name="Full Model",
            boxpoints="all",
            marker=dict(
                color=COLOR_PALETTE[-4],
                opacity=0.25,
            ),
            showlegend=False,
        ),
        row=1,
        col=i + 1,
    )


fig.update_yaxes(title_text="Answer Probability", row=1, col=1)
fig.update_layout(width=700, height=600)
fig.show()
folder: Path = repo_path_to_abs_path("figures/figures-12")
# Save figure as pdf in figures folder
fig.write_image(str(folder / "docstring-probability.pdf"))
fig.write_image(str(folder / "docstring-probability.svg"))
fig.write_image(str(folder / "docstring-probability.png"), scale=4)

#%%
