import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as t
import torch.utils.data
from attr import dataclass
from torch.utils.data import (
    DataLoader,
    Dataset,
)


@dataclass(frozen=True)
class PromptPair:
    clean: t.Tensor
    corrupt: t.Tensor
    answers: t.Tensor
    wrong_answers: t.Tensor


@dataclass(frozen=True)
class PromptPairBatch:
    key: int
    diverge_idx: int
    clean: t.Tensor
    corrupt: t.Tensor
    answers: List[t.Tensor] | t.Tensor
    wrong_answers: List[t.Tensor] | t.Tensor


def collate_fn(batch: List[PromptPair]) -> PromptPairBatch:
    clean = t.stack([p.clean for p in batch])
    corrupt = t.stack([p.corrupt for p in batch])
    if all([p.answers.shape == batch[0].answers.shape for p in batch]):
        answers = t.stack([p.answers for p in batch])
    else:  # Sometimes each prompt has a different number of answers
        answers = [p.answers for p in batch]
    if all([p.wrong_answers.shape == batch[0].wrong_answers.shape for p in batch]):
        wrong_answers = t.stack([p.wrong_answers for p in batch])
    else:  # Sometimes each prompt has a different number of wrong answers
        wrong_answers = [p.wrong_answers for p in batch]
    key = hash((str(clean.tolist()), str(corrupt.tolist())))

    diverge_idxs = (~(clean == corrupt)).int().argmax(dim=1)
    diverge_idx: int = int(diverge_idxs.min().item())
    return PromptPairBatch(key, diverge_idx, clean, corrupt, answers, wrong_answers)


class PromptDataset(Dataset):
    def __init__(
        self,
        clean_prompts: List[t.Tensor] | t.Tensor,
        corrupt_prompts: List[t.Tensor] | t.Tensor,
        answers: List[t.Tensor],
        wrong_answers: List[t.Tensor],
        seq_labels: Optional[List[str]] = None,
    ):
        self.clean_prompts = clean_prompts
        self.corrupt_prompts = corrupt_prompts
        self.answers = answers
        self.wrong_answers = wrong_answers
        self.seq_labels = seq_labels

    def __len__(self) -> int:
        assert len(self.clean_prompts) == len(self.corrupt_prompts)
        return len(self.clean_prompts)

    def __getitem__(self, idx: int) -> PromptPair:
        return PromptPair(
            self.clean_prompts[idx],
            self.corrupt_prompts[idx],
            self.answers[idx],
            self.wrong_answers[idx],
        )


class PromptDataLoader(DataLoader[PromptPairBatch]):
    def __init__(
        self,
        *args: Any,
        seq_len: Optional[int],
        seq_labels: Optional[List[str]] = None,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len
        self.seq_labels = seq_labels


def load_datasets_from_json(
    tokenizer: Any,
    path: Path,
    device: str,
    prepend_bos: bool = True,
    batch_size: int = 32,
    train_test_split: Sequence[int | float] = [0.9, 0.1],
    length_limit: int = 100,
    return_seq_length: bool = False,
    random_subet: bool = True,
    pad: bool = True,
) -> Tuple[PromptDataLoader, PromptDataLoader]:
    """Load a dataset from a json file. The file should specify a list of
    dictionaries with keys "clean_prompt" and "corrupt_prompt"."""
    with open(path, "r") as f:
        data = json.load(f)
    random.shuffle(data["prompts"]) if random_subet else None
    clean_prompts = [d["clean"] for d in data["prompts"]][:length_limit]
    corrupt_prompts = [d["corrupt"] for d in data["prompts"]][:length_limit]
    answer_strs = [d["answers"] for d in data["prompts"]][:length_limit]
    wrong_answer_strs = [d["wrong_answers"] for d in data["prompts"]][:length_limit]
    seq_labels = data.get("seq_labels", None)
    if tokenizer is None:
        clean_prompts = [t.tensor(p).to(device) for p in clean_prompts]
        corrupt_prompts = [t.tensor(p).to(device) for p in corrupt_prompts]
        answers = [t.tensor(a).to(device) for a in answer_strs]
        wrong_answers = [t.tensor(a).to(device) for a in wrong_answer_strs]
        seq_len = None
        assert return_seq_length is False
    else:
        if prepend_bos:
            clean_prompts = [tokenizer.bos_token + prompt for prompt in clean_prompts]
            corrupt_prompts = [
                tokenizer.bos_token + prompt for prompt in corrupt_prompts
            ]
        tokenizer.padding_side = "left"
        clean_prompts = tokenizer(clean_prompts, padding=pad, return_tensors="pt")
        corrupt_prompts = tokenizer(corrupt_prompts, padding=pad, return_tensors="pt")
        seq_len = None
        if return_seq_length:
            assert t.all(clean_prompts["attention_mask"] == 1)
            assert t.all(corrupt_prompts["attention_mask"] == 1)
            seq_len = clean_prompts["input_ids"].shape[1]
        ans_dict: List[Dict] = [tokenizer(a, return_tensors="pt") for a in answer_strs]
        wrong_ans_dict: List[Dict] = [
            tokenizer(a, return_tensors="pt") for a in wrong_answer_strs
        ]
        clean_prompts = clean_prompts["input_ids"].to(device)
        corrupt_prompts = corrupt_prompts["input_ids"].to(device)
        answers = [a["input_ids"].squeeze(-1).to(device) for a in ans_dict]
        wrong_answers = [a["input_ids"].squeeze(-1).to(device) for a in wrong_ans_dict]

    dataset = PromptDataset(clean_prompts, corrupt_prompts, answers, wrong_answers)
    train_set, test_set = torch.utils.data.random_split(dataset, train_test_split)
    train_loader = PromptDataLoader(
        train_set,
        seq_len=seq_len,
        seq_labels=seq_labels,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = PromptDataLoader(
        test_set,
        seq_len=seq_len,
        seq_labels=seq_labels,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader
