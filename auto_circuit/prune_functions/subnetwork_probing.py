# Some of this code is copied from:
# https://github.com/stevenxcao/subnetwork-probing
# From the paper Low-Complexity Probing via Finding Subnetworks:
# https://arxiv.org/abs/2104.03514
from typing import Any, Dict, List, Set
import plotly.express as px
from plotly import graph_objects as go

import torch as t
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, kl_div

from auto_circuit.data import PromptPairBatch
from auto_circuit.types import Edge, EdgeSrc
from auto_circuit.utils.graph_utils import graph_edges
from auto_circuit.utils.custom_tqdm import tqdm
from functools import reduce
import math

from auto_circuit.utils.misc import module_by_name, set_module_by_name

# Constants are copied from the paper's code
mask_p, left, right, temp = 0.9, -0.1, 1.1, 2 / 3

class MaskedModule(t.nn.Module):
    """A module that applies a mask to its output."""
    def __init__(self, module: t.nn.Module):
        super().__init__()
        self.masked = module
        self.masks: Dict[EdgeSrc, t.nn.Parameter] = {}

    def sample_mask(self, mask: t.Tensor) -> t.Tensor:
        u = t.zeros_like(mask).uniform_().clamp(0.0001, 0.9999)
        s = t.sigmoid((u.log() - (1 - u).log() + mask) / temp)
        s_bar = s * (right - left) + left
        return s_bar.clamp(min=0.0, max=1.0)

    def forward(self, *args: Any, **kwargs: Any) -> t.Tensor:
        out = self.masked(*args, **kwargs)
        out_mask = t.ones_like(out)
        for edge_src, mask in self.masks.items():
            out_mask[edge_src.t_idx] *= self.sample_mask(mask)
        return out * out_mask


def subnetwork_probing_prune_scores(
    model: t.nn.Module,
    factorized: bool,
    train_data: DataLoader[PromptPairBatch],
    learning_rate: float = 0.1,
    epochs: int = 20,
    max_lambda: float = 10,
) -> Dict[Edge, float]:
    """Prune scores are the mean activation magnitude of each edge."""
    edges = graph_edges(model, factorized)
    prune_scores = {}

    default_logprobs: Dict[str, t.Tensor] = {}
    with t.inference_mode():
        for batch in train_data:
            default_logprobs[batch.key] = log_softmax(model(batch.clean), dim=-1)
        # default_logprobs = [log_softmax(model(batch.clean), dim=-1) for batch in train_data]

    mask_modules: Dict[Edge, MaskedModule] = {}
    for edge in edges:
        if edge.src.module_name not in mask_modules:
            mod = edge.src.module(model)
            mask_mod = MaskedModule(mod)
            mask_modules[edge] = mask_mod
            set_module_by_name(model, edge.src.module_name, mask_mod)
        mask_mod = mask_modules[edge]
        p = (mask_p - left) / (right - left)
        init_mask_val=t.tensor(math.log(p / (1 - p)))
        mask_mod.masks[edge.src] = t.nn.Parameter(init_mask_val)
        
    mask_params: List[t.nn.Parameter] = [mask for (_, mod) in mask_modules.items()
                                            for (_, mask) in mod.masks.items()]
    optim = t.optim.Adam(mask_params, lr=learning_rate) # type: ignore
    loss_history, kl_div_history, regularize_history = [], [], []
    # for epoch in (epoch_pbar:=tqdm(range(epochs))):
    for epoch in tqdm(range(epochs)):
        # epoch_pbar.set_description_str(f"Subnetwork Probing Epoch {epoch}")
        lmbd = min(1, max(0, ((epoch - (0.25 * epochs)) / (0.5 * epochs)))) * max_lambda
        for batch_idx, batch in (batch_pbar:=tqdm(enumerate(train_data))):
            batch_pbar.set_description_str(f"Subnetwork Probing Batch {batch_idx}")
            masked_logprobs = log_softmax(model(batch.clean), dim=-1)
            kl_div_term = kl_div(masked_logprobs,
                                default_logprobs[batch.key],
                                reduction="batchmean",
                                log_target=True)
            masks: t.Tensor = t.stack(mask_params) # type: ignore
            regularize_term = t.sigmoid(masks - temp * math.log(-left / right)).mean()
            loss = kl_div_term + regularize_term * lmbd
            loss_history.append(loss.item())
            kl_div_history.append(kl_div_term.item())
            regularize_history.append(regularize_term.item() * lmbd)
            loss.backward()
            optim.step()
            model.zero_grad()

    for edge in edges:
        print(f"Edge {edge}, mask score", mask_modules[edge].masks[edge.src].mean().item()) # type: ignore
    
    # Plot loss , KL divergence , and regularization history on one graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, name="Loss"))
    fig.add_trace(go.Scatter(y=kl_div_history, name="KL Divergence"))
    fig.add_trace(go.Scatter(y=regularize_history, name="Regularization"))
    fig.update_layout(title="Subnetwork Probing Loss History",
                        xaxis_title="Iteration",
                        yaxis_title="Loss")
    fig.show()

    for edge in edges:
        prune_scores[edge] = mask_modules[edge].masks[edge.src].mean().item() # type: ignore
    return prune_scores
