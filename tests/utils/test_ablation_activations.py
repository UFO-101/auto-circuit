#%%
import torch as t

from auto_circuit.data import PromptDataLoader, PromptDataset
from auto_circuit.model_utils.micro_model_utils import Block
from auto_circuit.types import AblationType
from auto_circuit.utils.ablation_activations import src_ablations
from auto_circuit.utils.graph_utils import patchable_model
from tests.conftest import DEVICE

clean_inputs: t.Tensor = t.tensor(
    [
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ],
        [
            [13.0, 14.0],
            [15.0, 16.0],
            [17.0, 18.0],
        ],
        [
            [19.0, 20.0],
            [21.0, 22.0],
            [23.0, 24.0],
        ],
    ],
    device=DEVICE,
)  # shape: (batch=4, seq=3, resid=2)

micro_model_block = Block().to(DEVICE)
head_0_weight = micro_model_block.weights[0]
head_1_weight = micro_model_block.weights[1]


def micro_model_src_outs_stack(input_batch: t.Tensor) -> t.Tensor:
    """
    Manually calculate the activations of the micro model at the 5 SrcNodes.
    """
    resid_start_act = input_batch
    lyr_0_head_0_act = head_0_weight * resid_start_act
    lyr_0_head_1_act = head_1_weight * resid_start_act
    lyr_0_out = micro_model_block(input_batch) + input_batch
    lyr_1_head_0_act = head_0_weight * lyr_0_out
    lyr_1_head_1_act = head_1_weight * lyr_0_out
    micro_model_acts = t.stack(
        [
            resid_start_act,
            lyr_0_head_0_act,
            lyr_0_head_1_act,
            lyr_1_head_0_act,
            lyr_1_head_1_act,
        ]
    )
    return micro_model_acts


def test_resample_and_zero_src_ablations(micro_model: t.nn.Module):
    """
    Check that src_ablations returns the same activations as our manual calculation for
    the Resample and Zero AblationTypes.
    """
    model = patchable_model(
        micro_model, factorized=True, seq_len=3, separate_qkv=False, device=DEVICE
    )
    input_batch = clean_inputs

    resample_acts: t.Tensor = src_ablations(model, input_batch, AblationType.RESAMPLE)
    micro_model_acts = micro_model_src_outs_stack(input_batch)
    assert t.allclose(resample_acts, micro_model_acts)

    zero_acts: t.Tensor = src_ablations(model, input_batch, AblationType.ZERO)
    assert t.allclose(zero_acts, t.zeros_like(micro_model_acts))


def test_tokenwise_mean_src_ablations(micro_model: t.nn.Module):
    """
    Check that src_ablations returns the mean activations of the SrcNodes for the mean
    AblationTypes.
    """
    batch_size = 2
    model = patchable_model(
        micro_model, factorized=True, seq_len=3, separate_qkv=False, device=DEVICE
    )
    corrupt_inputs = t.rand_like(clean_inputs) * 5.0
    n_prompt, n_seq, n_vocab = clean_inputs.shape
    ans = [t.tensor([i % n_vocab], device=DEVICE) for i in range(n_prompt)]
    wrong_ans = [t.tensor([(i + 1) % n_vocab], device=DEVICE) for i in range(n_prompt)]
    micro_dataset = PromptDataset(clean_inputs, corrupt_inputs, ans, wrong_ans)
    micro_dataloader = PromptDataLoader(
        micro_dataset, seq_len=3, diverge_idx=0, batch_size=batch_size
    )

    tokenwise_mean_clean_ablations = src_ablations(
        model, micro_dataloader, AblationType.TOKENWISE_MEAN_CLEAN
    )
    src_outs = micro_model_src_outs_stack(clean_inputs)
    mean_clean_src_outs = src_outs.mean(dim=1, keepdim=True).repeat(1, batch_size, 1, 1)
    assert t.allclose(tokenwise_mean_clean_ablations, mean_clean_src_outs)

    tokenwise_mean_corrupt_ablations = src_ablations(
        model, micro_dataloader, AblationType.TOKENWISE_MEAN_CORRUPT
    )
    src_outs = micro_model_src_outs_stack(corrupt_inputs)
    mean_corr_src_outs = src_outs.mean(dim=1, keepdim=True).repeat(1, batch_size, 1, 1)
    assert t.allclose(tokenwise_mean_corrupt_ablations, mean_corr_src_outs)

    tokenwise_mean_clean_corr_ablations = src_ablations(
        model, micro_dataloader, AblationType.TOKENWISE_MEAN_CLEAN_AND_CORRUPT
    )
    mean_clean_corrupt_src_outs = (mean_clean_src_outs + mean_corr_src_outs) / 2
    assert t.allclose(tokenwise_mean_clean_corr_ablations, mean_clean_corrupt_src_outs)


# micro_model = micro_model()
# test_resample_and_zero_src_ablations(micro_model)
# test_tokenwise_mean_src_ablations(micro_model)
