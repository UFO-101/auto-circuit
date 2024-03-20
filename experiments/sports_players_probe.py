#%%
from typing import Dict

import torch as t

from auto_circuit.data import BatchKey, load_datasets_from_json
from auto_circuit.experiment_utils import (
    load_tl_model,
)
from auto_circuit.metrics.official_circuits.circuits.sports_players_official import (
    sports_players_true_edges,
)
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import batch_src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model, set_all_masks
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.utils.tensor_ops import (
    correct_answer_greater_than_incorrect_proportion,
)

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
model = load_tl_model("pythia-2.8b-deduped", device)
#%%

path = repo_path_to_abs_path(
    "datasets/sports-players/sports_players_pythia-2.8b-deduped_prompts.json"
)
_, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=1,
    train_test_size=(0, 1),
    shuffle=False,
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

sports_players_edges = sports_players_true_edges(
    model,
    word_idxs=test_loader.word_idxs,
    token_positions=True,
    seq_start_idx=test_loader.diverge_idx,
)
set_all_masks(model, 1.0)
for edge in sports_players_edges:
    edge.patch_mask(model).data[edge.patch_idx] = 0.0

patches: Dict[BatchKey, t.Tensor] = batch_src_ablations(
    model=model,
    dataloader=test_loader,
    ablation_type=AblationType.RESAMPLE,
    clean_corrupt="corrupt",
)
for batch in test_loader:
    with patch_mode(model, patch_src_outs=patches[batch.key]):
        patched_out = model(batch.clean)
        correct = correct_answer_greater_than_incorrect_proportion(patched_out, batch)
        print("Correct percent", correct.item())
