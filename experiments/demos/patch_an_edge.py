#%%
import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import AblationType, PruneScores
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model
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
    return_seq_length=True,
    tail_divergence=True,
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    seq_len=test_loader.seq_len,
    separate_qkv=True,
    kv_caches=(train_loader.kv_cache, test_loader.kv_cache),
    device=device,
)

ablations = src_ablations(model, test_loader, AblationType.TOKENWISE_MEAN_CORRUPT)

patch_edges = [
    "Resid Start->MLP 2",
    "MLP 2->A2.4.Q",
    "A2.4->Resid End",
]
with patch_mode(model, ablations, patch_edges):
    for batch in test_loader:
        patched_out = model(batch.clean)


attrution_patching_scores: PruneScores = mask_gradient_prune_scores(
    model=model,
    dataloader=test_loader,
    official_edges=None,
    grad_function="logit",
    answer_function="avg_diff",
)
