#%%
import torch as t

from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patch_mode, patchable_model

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = load_tl_model("gpt2", device)

tokens = model.to_tokens("Hello, world!")

model = patchable_model(
    model, factorized=True, slice_output="last_seq", separate_qkv=True, device=device
)

ablations = src_ablations(model, tokens, AblationType.ZERO)

patch_edges = [
    "Resid Start->MLP 2",
    "MLP 2->A2.4.Q",
    "A2.4->Resid End",
]
with patch_mode(model, ablations, patch_edges):
    patched_out = model(tokens)
