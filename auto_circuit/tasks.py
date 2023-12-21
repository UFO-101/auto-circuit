from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch as t
import transformer_lens as tl

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.metrics.official_circuits.docstring_official import (
    docstring_true_edges,
)
from auto_circuit.metrics.official_circuits.greaterthan_official import (
    greaterthan_true_edges,
)
from auto_circuit.metrics.official_circuits.ioi_official import ioi_true_edges
from auto_circuit.types import Edge, TaskKey
from auto_circuit.utils.graph_utils import prepare_model
from auto_circuit.utils.misc import repo_path_to_abs_path

MODEL_CACHE: Dict[str, t.nn.Module] = {}
DATASET_CACHE: Dict[str, Tuple[PromptDataLoader, PromptDataLoader]] = {}


@dataclass
class Task:
    key: TaskKey
    name: str
    batch_size: int
    batch_count: int
    token_circuit: bool
    _model_def: str | t.nn.Module
    _dataset_name: str
    _true_edge_func: Optional[Callable[..., Set[Edge]]] = None
    __init_complete__: bool = False

    @property
    def true_edges(self) -> Set[Edge]:
        if self._true_edge_func is None:
            raise ValueError("This task does not have a true edge function")
        self.init_task() if not self.__init_complete__ else None
        return self._true_edges

    @property
    def model(self) -> t.nn.Module:
        self.init_task() if not self.__init_complete__ else None
        return self._model

    @property
    def train_loader(self) -> PromptDataLoader:
        self.init_task() if not self.__init_complete__ else None
        return self._train_loader

    @property
    def test_loader(self) -> PromptDataLoader:
        self.init_task() if not self.__init_complete__ else None
        return self._test_loader

    def init_task(self):
        if isinstance(self._model_def, t.nn.Module):
            self._model = self._model_def
            self.device: t.device = next(self._model.parameters()).device
        elif self._model_def in MODEL_CACHE:
            self._model = MODEL_CACHE[self._model_def]
            self.device: t.device = next(self._model.parameters()).device
        else:
            device_str = "cuda" if t.cuda.is_available() else "cpu"
            self.device: t.device = t.device(device_str)
            model = tl.HookedTransformer.from_pretrained(
                self._model_def, device=self.device
            )
            model.cfg.use_attn_result = True
            model.cfg.use_split_qkv_input = True
            model.cfg.use_hook_mlp_in = True
            assert model.tokenizer is not None
            model.eval()
            MODEL_CACHE[self._model_def] = model
            self._model = model

        if self._dataset_name in DATASET_CACHE:
            self._train_loader, self._test_loader = DATASET_CACHE[self._dataset_name]
        else:
            dataloader_len = self.batch_size * self.batch_count
            tknr = self._model.tokenizer if hasattr(self._model, "tokenizer") else None
            train_loader, test_loader = load_datasets_from_json(
                tokenizer=tknr,
                path=repo_path_to_abs_path(f"datasets/{self._dataset_name}.json"),
                device=self.device,
                prepend_bos=True,
                batch_size=self.batch_size,
                train_test_split=[dataloader_len, dataloader_len],
                length_limit=dataloader_len * 2,
                return_seq_length=self.token_circuit,
                pad=True,
            )
            DATASET_CACHE[self._dataset_name] = (train_loader, test_loader)
            self._train_loader, self._test_loader = (train_loader, test_loader)

        seq_len = self._train_loader.seq_len
        prepare_model(
            self._model,
            factorized=True,
            slice_output=True,
            seq_len=seq_len,
            device=self.device,
        )

        if self._true_edge_func is not None:
            if self.token_circuit:
                self._true_edges = self._true_edge_func(self._model, True)
            else:
                self._true_edges = self._true_edge_func(self._model)
        self.__init_complete__ = True


IOI_TOKEN_CIRCUIT_TASK: Task = Task(
    key="Indirect Object Identification Token Circuit",
    name="Indirect Object Identification",
    _model_def="gpt2-small",
    _dataset_name="ioi_single_template_prompts",
    batch_size=64,
    batch_count=2,
    _true_edge_func=ioi_true_edges,
    token_circuit=True,
)
IOI_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Indirect Object Identification Component Circuit",
    name="Indirect Object Identification",
    _model_def="gpt2-small",
    _dataset_name="ioi_prompts",
    batch_size=64,
    batch_count=2,
    _true_edge_func=ioi_true_edges,
    token_circuit=False,
)
DOCSTRING_TOKEN_CIRCUIT_TASK: Task = Task(
    key="Docstring Token Circuit",
    name="Docstring",
    _model_def="attn-only-4l",
    _dataset_name="docstring_prompts",
    batch_size=128,
    batch_count=2,
    _true_edge_func=docstring_true_edges,
    token_circuit=True,
)
DOCSTRING_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Docstring Component Circuit",
    name="Docstring",
    _model_def="attn-only-4l",
    _dataset_name="docstring_prompts",
    batch_size=128,
    batch_count=2,
    _true_edge_func=docstring_true_edges,
    token_circuit=False,
)
GREATERTHAN_COMPONENT_CIRCUIT_TASK: Task = Task(
    key="Greaterthan Component Circuit",
    name="Greaterthan",
    _model_def="gpt2-small",
    _dataset_name="greaterthan_gpt2-small_prompts",
    batch_size=64,
    batch_count=2,
    _true_edge_func=greaterthan_true_edges,
    token_circuit=False,
)

TASKS: List[Task] = [
    IOI_TOKEN_CIRCUIT_TASK,
    IOI_COMPONENT_CIRCUIT_TASK,
    DOCSTRING_TOKEN_CIRCUIT_TASK,
    DOCSTRING_COMPONENT_CIRCUIT_TASK,
    GREATERTHAN_COMPONENT_CIRCUIT_TASK,
]
TASK_DICT: Dict[TaskKey, Task] = {task.key: task for task in TASKS}
