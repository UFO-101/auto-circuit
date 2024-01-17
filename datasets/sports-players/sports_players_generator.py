#%%
import json
import random

import torch as t
import transformer_lens as tl

from auto_circuit.utils.misc import repo_path_to_abs_path


def read_players(filename: str):
    filepath = repo_path_to_abs_path(f"datasets/sports-players/{filename}")
    with open(filepath, "r") as file:
        players = file.readlines()
    return [player.strip() for player in players]


american_football_players = read_players("american-football-players.txt")
basketball_players = read_players("basketball-players.txt")
baseball_players = read_players("baseball-players.txt")

template = "Fact: Tiger Woods plays the sport of golf\nFact: {} plays the sport of"

football_prompts = [template.format(player) for player in american_football_players]
basketball_prompts = [template.format(player) for player in basketball_players]
baseball_prompts = [template.format(player) for player in baseball_players]

MODEL_NAME = "pythia-2.8b-deduped"

device = "cuda" if t.cuda.is_available() else "cpu"
model = tl.HookedTransformer.from_pretrained(MODEL_NAME, device=device)

football_valid_idxs, basketball_valid_idxs, baseball_valid_idxs = [], [], []

for prompts, answer, valid_idxs in [
    (football_prompts, " football", football_valid_idxs),
    (basketball_prompts, " basketball", basketball_valid_idxs),
    (baseball_prompts, " baseball", baseball_valid_idxs),
]:
    ans_tok = model.to_tokens(answer, padding_side="left", prepend_bos=False)[0][0]
    prompt_tokens = model.to_tokens(prompts, prepend_bos=True, padding_side="left")
    print("prompt_tokens.shape", prompt_tokens.shape)

    correct_prompt_len = 19
    padding_token = model.tokenizer.pad_token_id  # type: ignore
    correct_prompt_len_idxs = t.where(
        # <pad> == <bos> so we need to subtract 1 from the correct prompt len
        (prompt_tokens != padding_token).sum(dim=1)
        == correct_prompt_len - 1
    )[0]
    prompt_tokens = prompt_tokens[correct_prompt_len_idxs, -correct_prompt_len:]
    print("prompt_tokens.shape", prompt_tokens.shape)

    with t.inference_mode():
        logits = model(prompt_tokens)[:, -1]
    probs = t.softmax(logits, dim=-1)
    correct_answer_idxs = t.where(probs[:, ans_tok] > 0.5)[0]

    final_idxs = t.arange(len(prompts), device=device)[correct_prompt_len_idxs][
        correct_answer_idxs
    ]
    print("final_idxs.shape", final_idxs.shape)
    valid_idxs.extend(final_idxs.tolist())

min_valid_idxs = min(
    len(football_valid_idxs), len(basketball_valid_idxs), len(baseball_valid_idxs)
)

prompt_dicts = []

sport_objects = [
    [" football", football_valid_idxs, football_prompts],
    [" basketball", basketball_valid_idxs, basketball_prompts],
    [" baseball", baseball_valid_idxs, baseball_prompts],
]
for sport_idx in range(len(sport_objects)):
    # Shuffle the valid_idxs
    for valid_idxs in [football_valid_idxs, basketball_valid_idxs, baseball_valid_idxs]:
        random.shuffle(valid_idxs)
    correct_answer = sport_objects[sport_idx][0]
    incorrect_answers = [
        sport_objects[idx][0] for idx in range(len(sport_objects)) if idx != sport_idx
    ]
    clean_valid_idxs = sport_objects[sport_idx][1]
    clean_prompts = sport_objects[sport_idx][2]
    for i in range(min_valid_idxs):
        # Randomly choose a different sport
        rand_sport_idx = random.choice(
            [idx for idx in range(len(sport_objects)) if idx != sport_idx]
        )
        corrupt_prompts = sport_objects[rand_sport_idx][2]
        corrupt_valid_idxs = sport_objects[rand_sport_idx][1]
        prompt_dict = {
            "clean": clean_prompts[clean_valid_idxs[i]],
            "corrupt": corrupt_prompts[corrupt_valid_idxs[i]],
            "answers": [correct_answer],
            "wrong_answers": incorrect_answers,
        }
        prompt_dicts.append(prompt_dict)

data_json = {"prompts": prompt_dicts}

with open(f"sports_players_{MODEL_NAME}_prompts.json", "w") as f:
    json.dump(data_json, f)
#%%
