#%%
import random
from typing import Dict

import pytest
import transformer_lens as tl
from typing_extensions import Literal

from auto_circuit.experiment_utils import (
    ioi_circuit_single_template_logit_diff_percent,
)
from auto_circuit.types import AblationType

"""
This dictionary shows the expected logit different percentage when mean ablating the
compliment of the IOI heads ** with 50 samples **. The keys correspond to
[prepend_bos, prompt_format, prompt_idx].
These results can be reproduced using my fork of the Redwood Research open-source IOI
implementation: https://github.com/UFO-101/IOI, at the top of experiments.py

!!! Note: Different sample counts give different results. !!!
"""
IOI_TRUE_RESULTS: Dict[bool, Dict[str, Dict[int, float]]] = {
    True: {
        "ABBA": {
            0: 89.93476867675781,
            1: 83.50794982910156,
            2: 91.7087631225586,
            3: 98.55170440673828,
            4: 79.78076171875,
            5: 89.12063598632812,
            6: 77.12073516845703,
            7: 74.34590148925781,
            8: 96.33061218261719,
            9: 91.64030456542969,
            10: 93.10379791259766,
            11: 94.52410125732422,
            12: 79.83614349365234,
            13: 78.6539535522461,
            14: 75.51397705078125,
        },
        "BABA": {
            0: 121.23332214355469,
            1: 114.96614074707031,
            2: 119.44014739990234,
            3: 131.1682891845703,
            4: 154.6707305908203,
            5: 137.20114135742188,
            6: 107.37574005126953,
            7: 113.95597839355469,
            8: 133.82261657714844,
            9: 136.47177124023438,
            10: 107.77767944335938,
            11: 123.02869415283203,
            12: 146.77886962890625,
            13: 122.05414581298828,
            14: 112.42205047607422,
        },
    },
    False: {
        "ABBA": {
            0: 86.49168395996094,
            1: 80.72254943847656,
            2: 83.31563568115234,
            3: 91.4164047241211,
            4: 75.28430938720703,
            5: 81.16667175292969,
            6: 61.94364929199219,
            7: 57.411338806152344,
            8: 84.03655242919922,
            9: 79.22144317626953,
            10: 84.04419708251953,
            11: 89.66465759277344,
            12: 76.80400848388672,
            13: 81.1402587890625,
            14: 72.38204193115234,
        },
        "BABA": {
            0: 115.64643096923828,
            1: 110.63795471191406,
            2: 114.98576354980469,
            3: 123.69253540039062,
            4: 142.0835418701172,
            5: 124.64212036132812,
            6: 101.43853759765625,
            7: 108.3462142944336,
            8: 132.2772979736328,
            9: 135.2960662841797,
            10: 98.9802017211914,
            11: 119.48235321044922,
            12: 144.59426879882812,
            13: 112.72322082519531,
            14: 100.47361755371094,
        },
    },
}


def test_ioi_faithfulness_exact_repro(
    gpt2: tl.HookedTransformer, random_seed: int = 0, debug: bool = False
):
    """
    These tests check that we can exactly reproduce the faithfulness results of the IOI
    paper. They compute the logit different percentage when ablating all heads except
    those in the IOI circuit with the mean ablation calculated over a sample of the ABC
    distribution.
    """
    # randomly choose values for prepend_bos, template, template_idx and factorized
    random.seed(random_seed)
    prepend_bos: bool = random.choice([True, False])
    template: Literal["ABBA", "BABA"] = random.choice(["ABBA", "BABA"])
    factorized: bool = random.choice([True, False])

    template_idx: int = random.choice(range(15))
    expected_logit_diff_percent = IOI_TRUE_RESULTS[prepend_bos][template][template_idx]
    actual_logit_diff_percent = ioi_circuit_single_template_logit_diff_percent(
        gpt2,
        test_batch_size=50,
        prepend_bos=prepend_bos,
        template=template,
        template_idx=template_idx,
        factorized=factorized,
        true_circuit="Nodes",
        ablation_type=AblationType.TOKENWISE_MEAN_CORRUPT,
    )
    # compare the expected and actual logit different percentages
    if debug:
        print(
            "prepend_bos:",
            prepend_bos,
            "template:",
            template,
            "template_idx:",
            template_idx,
        )
        print("actual_logit_diff_percent:", actual_logit_diff_percent)
        print("expected_logit_diff_percent:", expected_logit_diff_percent)
    assert actual_logit_diff_percent == pytest.approx(
        expected_logit_diff_percent, abs=0.01
    )


# model = gpt2()
# for i in tqdm(range(20)):
#     test_ioi_faithfulness_exact_repro(model, random_seed=i, debug=False)
