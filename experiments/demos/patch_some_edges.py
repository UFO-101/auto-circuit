#%%
import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model
from auto_circuit.utils.misc import repo_path_to_abs_path
from auto_circuit.visualize import draw_seq_graph

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = load_tl_model("gpt2", device)

path = repo_path_to_abs_path("datasets/ioi/ioi_vanilla_template_prompts.json")
train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=16,
    train_test_size=(128, 128)
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=True,
    device=device,
)

ablations = src_ablations(model, test_loader, AblationType.TOKENWISE_MEAN_CORRUPT)

patch_edges = [
    "Resid Start->MLP 1",
    "MLP 1->MLP 2",
    "MLP 1->MLP 3",
    "MLP 2->A5.2.Q",
    "MLP 3->A5.2.Q",
    "A5.2->Resid End",
]

with patch_mode(model, ablations, patch_edges):
    for batch in test_loader:
        patched_out = model(batch.clean)

prune_scores = model.current_patch_masks_as_prune_scores()
fig = draw_seq_graph(model, prune_scores)
fig.write_image(repo_path_to_abs_path("docs/assets/Small_Circuit_Viz.png"), scale=4)
