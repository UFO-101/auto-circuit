#%%
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import plotly.graph_objects as go
import torch as t
from plotly import subplots

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import (
    load_tl_model,
)
from auto_circuit.metrics.official_circuits.circuits.docstring_official import (
    docstring_true_edges,
)
from auto_circuit.metrics.prune_metrics.correct_answer_percent import (
    measure_correct_ans_percent,
)
from auto_circuit.prune import run_circuits
from auto_circuit.prune_algos.circuit_probing import circuit_probing_prune_scores
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

#%%

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
model = load_tl_model("attn-only-4l", device)

path = repo_path_to_abs_path("datasets/docstring_prompts.json")
train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=128,
    train_test_size=(128, 128),
    shuffle=False,
    return_seq_length=True,
    tail_divergence=True,
)


#%%

# Mapping from [token_positions, edge_circuit] to the proportion of correct answers
results: Dict[AblationType, Dict[bool, float]] = defaultdict(dict)
default_avg_correct: float = 0.0
ablation_types: List[AblationType] = [
    AblationType.RESAMPLE,
    AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT,
]

for ablation_type in tqdm(ablation_types):
    for learned in tqdm([True, False]):
        patch_model = deepcopy(model)
        patch_model = patchable_model(
            model=patch_model,
            factorized=True,
            slice_output="last_seq",
            seq_len=test_loader.seq_len,
            separate_qkv=True,
            kv_caches=(test_loader.kv_cache,),
            device=device,
        )

        docstring_edges = docstring_true_edges(
            patch_model,
            word_idxs=test_loader.word_idxs,
            token_positions=True,
            seq_start_idx=test_loader.diverge_idx,
        )

        if learned:
            prune_scores = circuit_probing_prune_scores(
                model=patch_model,
                dataloader=test_loader,
                official_edges=docstring_edges,
                epochs=100,
                learning_rate=0.1,
            )
        else:
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
        measurements: Measurements = measure_correct_ans_percent(
            model=patch_model,
            dataloader=test_loader,
            pruned_outs=circuit_outs,
            out_of_correct_and_incorrect_answers=False,
        )
        assert len(measurements) == 2
        n_edge, avg_correct = measurements[0]
        results[ablation_type][learned] = avg_correct

        # Also get the score for the full model
        all_edges, full_model_avg_correct = measurements[1]
        assert all_edges == patch_model.n_edges
        default_avg_correct = full_model_avg_correct

        del patch_model
#%%


def bar_name(learned: bool) -> str:
    return "Learned" if learned else "Canonical"


fig = subplots.make_subplots(
    rows=1,
    cols=len(ablation_types),
    column_titles=[str(a) for a in ablation_types],
    shared_yaxes=True,
)
for i, ablation_type in enumerate(ablation_types):
    fig.add_trace(
        go.Bar(
            x=[bar_name(learned) for learned in results[ablation_type].keys()],
            y=list(results[ablation_type].values()),
            marker_color=COLOR_PALETTE,
            showlegend=False,
        ),
        row=1,
        col=i + 1,
    )
fig.add_hline(
    y=default_avg_correct,
    line_dash="dot",
    line_color="black",
    annotation_text="Full Model",
    annotation_position="top left",
    row=1,  # type: ignore
    col=1,  # type: ignore
)
fig.add_hline(
    y=58,
    line_dash="dot",
    line_color="black",
    annotation_text="Reported Faithfulness",
    annotation_position="bottom left",
    row=1,  # type: ignore
    col=1,  # type: ignore
)
fig.add_hline(
    y=default_avg_correct,
    line_dash="dot",
    line_color="black",
    row=1,  # type: ignore
    col=2,  # type: ignore
)
fig.add_hline(y=58, line_dash="dot", line_color="black", row=1, col=2)  # type: ignore
fig = fig.update_layout(
    yaxis_title="Correct Answer (%)",
    width=700,
    height=600,
)
fig.show()
folder: Path = repo_path_to_abs_path("figures/figures-12")
# Save figure as pdf in figures folder
# fig.write_image(str(folder / "docstring-faithfulness.pdf"))
# fig.write_image(str(folder / "docstring-faithfulness.svg"))
# fig.write_image(str(folder / "docstring-faithfulness.png"), scale=4)
# %%
