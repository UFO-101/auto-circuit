#%%
import os
import random
from typing import Dict, Set

import numpy as np
import torch as t
import torch.backends.mps
import transformer_lens as tl

import auto_circuit
import auto_circuit.data
import auto_circuit.prune
import auto_circuit.utils.graph_utils

# from auto_circuit.prune_functions.parameter_integrated_gradients import (
#     BaselineWeights,
# )
from auto_circuit.types import Edge, EdgeCounts, PatchType
from auto_circuit.utils.graph_utils import prepare_model
from auto_circuit.utils.misc import percent_gpu_mem_used, repo_path_to_abs_path

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
os.environ["TOKENIZERS_PARALLELISM"] = "False"
#%%

device = "cuda" if t.cuda.is_available() else "cpu"
print("device", device)
# model = tl.HookedTransformer.from_pretrained("gpt2-small", device=device)
model = tl.HookedTransformer.from_pretrained("attn-only-4l", device=device)
# model = tl.HookedTransformer.from_pretrained("tiny-stories-33M", device=device)

model.cfg.use_attn_result = True
model.cfg.use_split_qkv_input = True
model.cfg.use_hook_mlp_in = True
assert model.tokenizer is not None
# model = t.compile(model)
model.eval()

# data_file = "datasets/indirect_object_identification.json"
# data_file = "datasets/greater_than_gpt2-small_prompts.json"
data_file = "datasets/docstring_prompts.json"
# data_file = "datasets/animal_diet_short_prompts.json"
# data_file = "datasets/mini_prompts.json"
print(percent_gpu_mem_used())

#%%
# ---------- Config ----------
experiment_type = PatchType.PATH_PATCH
factorized = True
# pig_baseline, pig_samples = BaselineWeights.ZERO, 50
default_edge_count_type = EdgeCounts.LOGARITHMIC
one_tao = 6e-2
acdc_tao_range, acdc_tao_step = (one_tao, one_tao), one_tao

train_loader, test_loader = auto_circuit.data.load_datasets_from_json(
    model.tokenizer,
    repo_path_to_abs_path(data_file),
    device=device,
    prepend_bos=True,
    batch_size=32,
    train_test_split=[0.5, 0.5],
    length_limit=64,
    pad=True,
)
prepare_model(
    model, factorized=factorized, slice_output=True, seq_len=None, device=device
)
edges: Set[Edge] = model.edges  # type: ignore
prune_scores_dict: Dict[str, Dict[Edge, float]] = {}
# ----------------------------

#%%
