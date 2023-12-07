#%%
import json
from typing import Dict, List


def single_input(multiple_answers: bool) -> Dict[str, List[List[float]] | List[int]]:
    return {
        "clean": [[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
        "corrupt": [[-3.0, -3.0], [-2.0, -2.0], [-1.0, -1.0]],
        "answers": [1] if not multiple_answers else [0, 1],
        "wrong_answers": [0] if not multiple_answers else [2, 3],
    }


MULTIPLE_ANSWERS = True
PROMPT_COUNT = 1000
prompts = [single_input(multiple_answers=MULTIPLE_ANSWERS) for _ in range(PROMPT_COUNT)]
data_json = {"prompts": prompts}

filename = f"micro_model_inputs{'_multiple_answers' if MULTIPLE_ANSWERS else ''}.json"
with open(filename, "w") as f:
    json.dump(data_json, f)
