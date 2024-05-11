#%%
import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
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
    train_test_size=(128, 128),
    return_seq_length=True
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    seq_len=test_loader.seq_len,
    separate_qkv=True,
    device=device,
)

attribution_scores: PruneScores = mask_gradient_prune_scores(
    model=model,
    dataloader=train_loader,
    official_edges=None,
    grad_function="logit",
    answer_function="avg_diff",
    mask_val=0.0,
)

fig = draw_seq_graph(model, attribution_scores, 3.5, seq_labels=train_loader.seq_labels)
fig.write_image(repo_path_to_abs_path("docs/assets/IOI_Tokenwise_Viz.png"), scale=4)
