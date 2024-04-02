#%%
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go
import torch as t
from plotly import subplots
from torch.nn.functional import softmax

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import (
    load_tl_model,
)
from auto_circuit.metrics.official_circuits.circuits.sports_players_official import (
    sports_players_true_edges,
)
from auto_circuit.metrics.prune_metrics.correct_answer_percent import (
    measure_correct_ans_percent,
)
from auto_circuit.prune import run_circuits
from auto_circuit.types import AblationType, CircuitOutputs, Measurements, PatchType
from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.graph_utils import (
    edge_counts_util,
    patchable_model,
)
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.tensor_ops import (
    indices_vals,
)

#%%
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
model = load_tl_model("pythia-2.8b-deduped", device)

path = repo_path_to_abs_path(
    "datasets/sports-players/new_sports_players_pythia-2.8b-deduped_prompts.json"
)
_, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=10,
    train_test_size=(0, 200),
    shuffle=True,
    return_seq_length=True,
    tail_divergence=True,
)

model = patchable_model(
    model=model,
    factorized=True,
    slice_output="last_seq",
    seq_len=test_loader.seq_len,
    separate_qkv=False,
    kv_caches=(test_loader.kv_cache,),
    device=device,
)

#%%
sports_players_edges = sports_players_true_edges(
    model,
    word_idxs=test_loader.word_idxs,
    token_positions=True,
    seq_start_idx=test_loader.diverge_idx,
)
prune_scores = model.circuit_prune_scores(sports_players_edges)

#%%
ablation_types: List[AblationType] = [
    AblationType.RESAMPLE,
    AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
]
out_of_correct_and_incorrect_answers: List[bool] = [True, False]
results: Dict[AblationType, Dict[bool, Measurements]] = defaultdict(dict)
probs: Dict[AblationType, Dict[int, t.Tensor]] = defaultdict(dict)
for ablation_type in tqdm(ablation_types):
    circuit_outs: CircuitOutputs = run_circuits(
        model=model,
        dataloader=test_loader,
        test_edge_counts=edge_counts_util(
            model.edges, prune_scores=prune_scores, all_edges=True, zero_edges=True
        ),
        prune_scores=prune_scores,
        patch_type=PatchType.TREE_PATCH,
        ablation_type=ablation_type,
    )
    for out_of_correct_and_incorrect_answer in out_of_correct_and_incorrect_answers:
        measurements: Measurements = measure_correct_ans_percent(
            model=model,
            dataloader=test_loader,
            pruned_outs=circuit_outs,
            out_of_correct_and_incorrect_answers=out_of_correct_and_incorrect_answer,
        )
        results[ablation_type][out_of_correct_and_incorrect_answer] = measurements

    for n_edges, batch_outs in circuit_outs.items():
        circ_probs = []
        for batch in test_loader:
            circ_batch_probs = softmax(batch_outs[batch.key], dim=-1)
            batch_probs = indices_vals(circ_batch_probs, batch.answers)
            circ_probs.append(batch_probs)
        n_edges_probs = t.cat(circ_probs, dim=0)
        probs[ablation_type][n_edges] = n_edges_probs

# %%
ablation_names = ["Resample", "Mean"]
fig = subplots.make_subplots(
    # rows=len(out_of_correct_and_incorrect_answers),
    # row_titles=["Sports Tokens", "All Tokens"],
    rows=1,
    cols=len(ablation_types),
    # column_titles=[str(ablation_type) for ablation_type in ablation_types],
    column_titles=ablation_names,
    shared_xaxes=True,
    # shared_yaxes=True,
    # y_title="Accuracy",
)
n_edge_names = ["Ablated Model", "Circuit", "Full Model"]
for i, ablation_type in enumerate(ablation_types):
    for j, out_of_correct_and_incorrect_answer in enumerate([True]):
        fig.add_trace(
            go.Bar(
                x=n_edge_names,
                y=[
                    m[1]
                    for m in results[ablation_type][out_of_correct_and_incorrect_answer]
                ],
                showlegend=False,
            ),
            row=j + 1,
            col=i + 1,
        )
        fig.update_yaxes(row=j + 1, col=i + 1, range=[-2, 105])
fig.update_yaxes(title_text="Accuracy", row=1, col=1)
margin = 20
fig.update_annotations(font_size=22)
fig.update_layout(
    width=600,
    height=400,
    margin=dict(l=margin * 4, r=margin, b=margin * 3.5, t=margin * 2),
)
fig.show()
folder: Path = repo_path_to_abs_path("figures/figures-12")
fig.write_image(str(folder / "sports-players-accuracy.pdf"))

fig = subplots.make_subplots(
    rows=1,
    cols=len(ablation_types),
    # column_titles=[str(ablation_type) for ablation_type in ablation_types],
    column_titles=ablation_names,
)
for i, (ablation_type, n_edge_probs) in enumerate(probs.items()):
    for n_edge_idx, (n_edge, points) in enumerate(n_edge_probs.items()):
        fig.add_trace(
            go.Box(
                y=points.flatten().tolist(),
                name=n_edge_names[n_edge_idx],
                showlegend=False,
            ),
            row=1,
            col=i + 1,
        )
        fig.update_yaxes(row=1, col=i + 1, range=[-0.02, 1.02])
fig.update_yaxes(title_text="Probability", row=1, col=1)
fig.update_annotations(font_size=22)
fig.update_layout(
    width=600,
    height=400,
    margin=dict(l=margin * 4, r=margin, b=margin * 3.5, t=margin * 2),
)
fig.show()
fig.write_image(str(folder / "sports-players-probability.pdf"))
