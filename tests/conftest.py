from typing import Any, Optional

import pytest
import torch as t
import transformer_lens as tl

from auto_circuit.data import PromptDataLoader, load_datasets_from_json
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.utils.misc import repo_path_to_abs_path

DEVICE = t.device("cuda") if t.cuda.is_available() else t.device("cpu")


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: Any, items: Any):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def mini_tl_transformer() -> tl.HookedTransformer:
    cfg = tl.HookedTransformerConfig(
        d_vocab=50257,
        n_layers=2,
        d_model=4,  # DON'T SET THIS TO 2 OR LAYERNORM WILL RUIN EVERYTHING
        n_ctx=64,
        n_heads=2,
        d_head=2,
        act_fn="gelu",
        tokenizer_name="gpt2",
        device=str(DEVICE),
    )
    mini_tl_model = tl.HookedTransformer(cfg)
    model = mini_tl_model
    model.init_weights()
    model.tokenizer.padding_side = "left"  # type: ignore
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    return model


def get_transformer(
    request: Optional[pytest.FixtureRequest], model_name: Optional[str] = None
) -> tl.HookedTransformer:
    name = request.param if request is not None else None
    if name is None:
        assert model_name is not None
        name = model_name
    model = tl.HookedTransformer.from_pretrained_no_processing(
        name, center_writing_weights=False, device=str(DEVICE)
    )
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    for param in model.parameters():
        param.requires_grad = False
    return model


@pytest.fixture(scope="session")
def hooked_transformer(
    request: pytest.FixtureRequest, model_name: Optional[str] = None
) -> tl.HookedTransformer:
    return get_transformer(request, model_name)


@pytest.fixture(scope="session")
def gpt2() -> tl.HookedTransformer:
    return get_transformer(None, "gpt2")


@pytest.fixture(scope="session")
def micro_dataloader(
    multiple_answers: bool = False, batch_count: int = 1, batch_size: int = 1
) -> PromptDataLoader:
    dataloader_len = batch_size * batch_count
    file_name = f"micro_model_inputs{'_multiple_answers' if multiple_answers else ''}"
    _, test_loader = load_datasets_from_json(
        None,
        repo_path_to_abs_path(f"datasets/{file_name}.json"),
        device=DEVICE,
        prepend_bos=False,
        batch_size=batch_size,
        train_test_size=(dataloader_len, dataloader_len),
    )
    return test_loader


@pytest.fixture(scope="session")
def micro_model() -> MicroModel:
    return MicroModel(n_layers=2).to(DEVICE)
