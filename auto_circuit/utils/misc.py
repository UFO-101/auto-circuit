import pickle
from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set

import torch as t
from einops import einsum
from torch.utils.hooks import RemovableHandle


def repo_path_to_abs_path(path: str) -> Path:
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path


def save_cache(data_dict: Dict[Any, Any], folder_name: str, base_filename: str):
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = folder / f"{base_filename}-{dt_string}.pkl"
    print(f"Saving cache to {file_path}")
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, f)


def load_cache(folder_name: str, filename: str) -> Dict[Any, Any]:
    folder = repo_path_to_abs_path(folder_name)
    with open(folder / filename, "rb") as f:
        return pickle.load(f)


@contextmanager
def remove_hooks() -> Iterator[Set[RemovableHandle]]:
    handles: Set[RemovableHandle] = set()
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def module_by_name(model: Any, module_name: str) -> t.nn.Module:
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    return reduce(getattr, init_mod + module_name.split("."))  # type: ignore


def set_module_by_name(model: Any, module_name: str, new_module: t.nn.Module) -> None:
    parent = model
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    if "." in module_name:
        parent = reduce(getattr, init_mod + module_name.split(".")[:-1])  # type: ignore
    setattr(parent, module_name.split(".")[-1], new_module)


def percent_gpu_mem_used(total_gpu_mib: int = 49000) -> str:
    return (
        "Memory used {:.1f}".format(
            ((t.cuda.memory_allocated() / (2**20)) / total_gpu_mib) * 100
        )
        + "%"
    )


def run_prompt(
    model: t.nn.Module,
    prompt: str,
    answer: Optional[str] = None,
    top_k: int = 10,
    prepend_bos: bool = False,
):
    print(" ")
    print("Testing prompt", model.to_str_tokens(prompt))
    toks = model.to_tokens(prompt, prepend_bos=prepend_bos)
    logits = model(toks)
    get_most_similar_embeddings(model, logits[0, -1], answer, top_k=top_k)


def get_most_similar_embeddings(
    model: t.nn.Module,
    out: t.Tensor,
    answer: Optional[str] = None,
    top_k: int = 10,
    apply_ln_final: bool = False,
    apply_unembed: bool = False,
    apply_embed: bool = False,
):
    assert not (apply_embed and apply_unembed), "Can't apply both embed and unembed"
    show_answer_rank = answer is not None
    answer = " cheese" if answer is None else answer
    out = out.unsqueeze(0).unsqueeze(0) if out.ndim == 1 else out
    out = model.ln_final(out) if apply_ln_final else out
    if apply_embed:
        unembeded = einsum(
            out, model.embed.W_E, "batch pos d_model, vocab d_model -> batch pos vocab"
        )
    elif apply_unembed:
        unembeded = model.unembed(out)
    else:
        unembeded = out
    answer_token = model.to_tokens(answer, prepend_bos=False).squeeze()
    answer_str_token = model.to_str_tokens(answer, prepend_bos=False)
    assert len(answer_str_token) == 1
    logits = unembeded.squeeze()  # type: ignore
    probs = logits.softmax(dim=-1)

    sorted_token_probs, sorted_token_values = probs.sort(descending=True)
    # Janky way to get the index of the token in the sorted list
    if answer is not None:
        correct_rank = t.arange(len(sorted_token_values))[
            (sorted_token_values == answer_token).cpu()
        ].item()
    else:
        correct_rank = -1
    if show_answer_rank:
        print(
            f'\n"{answer_str_token[0]}" token rank:',
            f"{correct_rank: <8}",
            f"\nLogit: {logits[answer_token].item():5.2f}",
            f"Prob: {probs[answer_token].item():6.2%}",
        )
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[sorted_token_values[i]].item():5.2f}",
            f"Prob: {sorted_token_probs[i].item():6.2%}",
            f'Token: "{model.to_string(sorted_token_values[i])}"',
        )
