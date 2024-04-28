#%%
import json
import random
from itertools import product
from typing import Any, Dict, Set

from auto_circuit.model_utils.tracr_model_utils import (
    BOS,
    MAX_SEQ_LEN,
    REVERSE_VOCAB,
    get_tracr_model,
)

_, tracr_model = get_tracr_model("reverse", "cpu")
model_encode = tracr_model.input_encoder.encode  # type: ignore


def generate_prompts(vocab: Set[Any], seq_len: int) -> Dict[str, Any]:
    # Generate all possible sequences of length SEQ_LEN - 1
    all_seqs = list(product(vocab, repeat=seq_len))

    prompts = []
    for seq_idx, clean_seq in enumerate(all_seqs):
        # Choose a different sequence. Make sure it's different to the clean sequence
        all_other_seqs = all_seqs[:seq_idx] + all_seqs[seq_idx + 1 :]
        corrupt_seq = random.choice(all_other_seqs)
        answer_seq = model_encode([BOS] + list(reversed(clean_seq)))[1:]
        wrong_answer_seq = model_encode([BOS] + list(reversed(corrupt_seq)))[1:]
        prompts.append(
            {
                "clean": model_encode([BOS] + list(clean_seq)),
                "corrupt": model_encode([BOS] + list(corrupt_seq)),
                "answers": [[x] for x in answer_seq],
                "wrong_answers": [[x] for x in wrong_answer_seq],
            }
        )
    return {"prompts": prompts}


with open(f"tracr_reverse_len_{MAX_SEQ_LEN}_prompts.json", "w") as f:
    json.dump(generate_prompts(REVERSE_VOCAB, MAX_SEQ_LEN), f)
