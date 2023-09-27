import json
from typing import Any, List, Sequence, Tuple

import torch as t
import torch.utils.data
from attr import dataclass
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PromptPair:
    clean: t.Tensor
    corrupt: t.Tensor


@dataclass(frozen=True)
class PromptPairBatch:
    key: int
    clean: t.Tensor
    corrupt: t.Tensor


def collate_fn(batch: List[PromptPair]) -> PromptPairBatch:
    clean = t.stack([p.clean for p in batch])
    corrupt = t.stack([p.corrupt for p in batch])
    key = hash((str(clean.tolist()), str(corrupt.tolist())))
    return PromptPairBatch(key, clean, corrupt)


class PromptDataset(Dataset):
    def __init__(self, clean_prompts: List[t.Tensor], corrupt_prompts: List[t.Tensor]):
        self.clean_prompts = clean_prompts
        self.corrupt_prompts = corrupt_prompts

    def __len__(self) -> int:
        assert len(self.clean_prompts) == len(self.corrupt_prompts)
        return len(self.clean_prompts)

    def __getitem__(self, idx: int) -> PromptPair:
        return PromptPair(self.clean_prompts[idx], self.corrupt_prompts[idx])


def load_datasets_from_json(
    tokenizer: Any,
    path: str,
    device: str,
    prepend_bos: bool = True,
    batch_size: int = 32,
    train_test_split: Sequence[int | float] = [0.9, 0.1],
    length_limit: int = 100,
) -> Tuple[DataLoader[PromptPairBatch], DataLoader[PromptPairBatch]]:
    """Load a dataset from a json file. The file should specify a list of
    dictionaries with keys "clean_prompt" and "corrupt_prompt"."""
    with open(path, "r") as f:
        data = json.load(f)
    clean_prompts = [d["clean"] for d in data["prompts"][:length_limit]]
    corrupt_prompts = [d["corrupt"] for d in data["prompts"][:length_limit]]
    if tokenizer is None:
        clean_prompts = [t.tensor(p).to(device) for p in clean_prompts]
        corrupt_prompts = [t.tensor(p).to(device) for p in corrupt_prompts]
    else:
        if prepend_bos:
            clean_prompts = [tokenizer.bos_token + prompt for prompt in clean_prompts]
            corrupt_prompts = [
                tokenizer.bos_token + prompt for prompt in corrupt_prompts
            ]
        tokenizer.padding_side = "left"
        clean_prompts = tokenizer(
            clean_prompts, padding=True, truncation=True, return_tensors="pt"
        )
        corrupt_prompts = tokenizer(
            corrupt_prompts, padding=True, truncation=True, return_tensors="pt"
        )
        clean_prompts = clean_prompts["input_ids"].to(device)
        corrupt_prompts = corrupt_prompts["input_ids"].to(device)
    dataset = PromptDataset(clean_prompts, corrupt_prompts)
    train_set, test_set = torch.utils.data.random_split(dataset, train_test_split)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, test_loader
