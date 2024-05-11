#%%
from typing import Dict

import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.metrics.prune_metrics.kl_div import measure_kl_div
from auto_circuit.prune import run_circuits
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    PatchType,
    PruneScores,
)
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import edge_counts_util, patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = load_tl_model("gpt2", device)

path = repo_path_to_abs_path("datasets/ioi/ioi_vanilla_template_prompts.json")
train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=16,
    train_test_size=(128, 128),
    tail_divergence=True,
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=True,
    kv_caches=(train_loader.kv_cache, test_loader.kv_cache),
    device=device,
)

ablations = src_ablations(model, test_loader, AblationType.RESAMPLE)

patch_edges: Dict[str, float] = {
    "Resid Start->MLP 1": 1.0,
    "MLP 1->MLP 2": 2.0,
    "MLP 1->MLP 3": 1.0,
    "MLP 2->A5.2.Q": 2.0,
    "MLP 3->A5.2.Q": 1.0,
    "A5.2->Resid End": 1.0,
}
ps: PruneScores = model.circuit_prune_scores(edge_dict=patch_edges)

circuit_outs: CircuitOutputs = run_circuits(
    model=model,
    dataloader=test_loader,
    test_edge_counts=edge_counts_util(model.edges, prune_scores=ps),
    prune_scores=ps,
    patch_type=PatchType.EDGE_PATCH,
    ablation_type=AblationType.RESAMPLE,
)

kl_divs = measure_kl_div(model, test_loader, circuit_outs)
