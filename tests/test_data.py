#%%
import torch as t
import transformer_lens as tl

from auto_circuit.data import load_datasets_from_json
from auto_circuit.utils.misc import repo_path_to_abs_path
from tests.conftest import DEVICE


def test_tail_divergence(gpt2: tl.HookedTransformer):
    dataset_path = repo_path_to_abs_path("datasets/ioi_single_template_prompts.json")
    _, kv_cache_test_loader = load_datasets_from_json(
        model=gpt2,
        path=dataset_path,
        device=DEVICE,
        batch_size=8,
        train_test_split=[0.5, 0.5],
        length_limit=16,
        return_seq_length=True,
        tail_divergence=True,
        random_subet=False,
    )
    kv_cache = kv_cache_test_loader.kv_cache
    diverge_idx = kv_cache_test_loader.diverge_idx

    _, no_cache_test_loader = load_datasets_from_json(
        model=gpt2,
        path=dataset_path,
        device=DEVICE,
        batch_size=8,
        train_test_split=[0.5, 0.5],
        length_limit=16,
        return_seq_length=True,
        tail_divergence=False,
        random_subet=False,
    )

    for kv_batch, no_cache_batch in zip(kv_cache_test_loader, no_cache_test_loader):
        assert kv_batch.clean.shape[0] == no_cache_batch.clean.shape[0]
        assert kv_batch.clean.shape[1] + diverge_idx == no_cache_batch.clean.shape[1]
        assert t.equal(kv_batch.clean, no_cache_batch.clean[:, diverge_idx:])
        kv_out = gpt2(kv_batch.clean, past_kv_cache=kv_cache)
        no_cache_out = gpt2(no_cache_batch.clean)
        assert kv_out.shape[0] == no_cache_out.shape[0]
        assert kv_out.shape[1] + diverge_idx == no_cache_out.shape[1]
        assert t.allclose(kv_out, no_cache_out[:, diverge_idx:], atol=1e-4)


def test_determinism_same_seed(gpt2: tl.HookedTransformer):
    dataset_path = repo_path_to_abs_path("datasets/ioi_single_template_prompts.json")
    train_loader_1, test_loader_1 = load_datasets_from_json(
        model=gpt2,
        path=dataset_path,
        device=DEVICE,
        batch_size=8,
        train_test_split=[0.5, 0.5],
        length_limit=16,
        return_seq_length=True,
        tail_divergence=False,
        random_subet=True,
        random_seed=0,
    )
    train_loader_2, test_loader_2 = load_datasets_from_json(
        model=gpt2,
        path=dataset_path,
        device=DEVICE,
        batch_size=8,
        train_test_split=[0.5, 0.5],
        length_limit=16,
        return_seq_length=True,
        tail_divergence=False,
        random_subet=True,
        random_seed=0,
    )
    for batch_1, batch_2 in zip(train_loader_1, train_loader_2):
        assert t.equal(batch_1.clean, batch_2.clean)
        assert t.equal(batch_1.corrupt, batch_2.corrupt)
        assert t.equal(batch_1.answers, batch_2.answers)
        assert t.equal(batch_1.wrong_answers, batch_2.wrong_answers)
    for batch_1, batch_2 in zip(test_loader_1, test_loader_2):
        assert t.equal(batch_1.clean, batch_2.clean)
        assert t.equal(batch_1.corrupt, batch_2.corrupt)
        assert t.equal(batch_1.answers, batch_2.answers)
        assert t.equal(batch_1.wrong_answers, batch_2.wrong_answers)


def test_determinism_different_seeds(gpt2: tl.HookedTransformer):
    dataset_path = repo_path_to_abs_path("datasets/ioi_single_template_prompts.json")
    train_loader_1, test_loader_1 = load_datasets_from_json(
        model=gpt2,
        path=dataset_path,
        device=DEVICE,
        batch_size=8,
        train_test_split=[0.5, 0.5],
        length_limit=16,
        return_seq_length=True,
        tail_divergence=False,
        random_subet=True,
        random_seed=0,
    )
    train_loader_2, test_loader_2 = load_datasets_from_json(
        model=gpt2,
        path=dataset_path,
        device=DEVICE,
        batch_size=8,
        train_test_split=[0.5, 0.5],
        length_limit=16,
        return_seq_length=True,
        tail_divergence=False,
        random_subet=True,
        random_seed=1,
    )
    for batch_1, batch_2 in zip(train_loader_1, train_loader_2):
        assert not t.equal(batch_1.clean, batch_2.clean)
        assert not t.equal(batch_1.corrupt, batch_2.corrupt)
        assert not t.equal(batch_1.answers, batch_2.answers)
        assert not t.equal(batch_1.wrong_answers, batch_2.wrong_answers)
    for batch_1, batch_2 in zip(test_loader_1, test_loader_2):
        assert not t.equal(batch_1.clean, batch_2.clean)
        assert not t.equal(batch_1.corrupt, batch_2.corrupt)
        assert not t.equal(batch_1.answers, batch_2.answers)
        assert not t.equal(batch_1.wrong_answers, batch_2.wrong_answers)


# model = gpt2()
# test_tail_divergence(model)
# test_determinism_same_seed(model)
# test_determinism_different_seeds(model)
