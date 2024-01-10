#%%
import json
import random

import requests
import torch as t
import transformer_lens as tl

url = "https://raw.githubusercontent.com/icyrockcom/country-capitals/master/data/country-list.json"
country_capital_data = requests.get(url).json()
country_to_captial = {}
for country_dict in country_capital_data:
    country_to_captial[country_dict["country"]] = country_dict["capital"]

TEMPLATE = "The capital of {country} is the city of"
MODEL_NAME = "pythia-70m-deduped"

passed_countries = []
model = tl.HookedTransformer.from_pretrained(MODEL_NAME, device="cpu")

prompts = [TEMPLATE.format(country=country) for country in country_to_captial]
answers = [" " + city for city in country_to_captial.values()]

answer_tokens = model.to_tokens(answers, padding_side="left", prepend_bos=False)

padding_token = model.tokenizer.pad_token_id  # type: ignore
padding_tokens = answer_tokens == padding_token

# Get idxs where the answer is a single token.
single_answer_token_idxs = t.where(
    padding_tokens.sum(dim=1) == answer_tokens.shape[1] - 1
)[0]

prompt_tokens = model.to_tokens(prompts, prepend_bos=True, padding_side="left")
prompt_tokens = prompt_tokens[single_answer_token_idxs]
min_prompt_len = (prompt_tokens != padding_token).sum(dim=1).min().item()
min_prompt_len_idxs = t.where(
    (prompt_tokens != padding_token).sum(dim=1) == min_prompt_len
)[0]

prompt_tokens = prompt_tokens[min_prompt_len_idxs]
logits = model(prompt_tokens)
max_tokens = logits[:, -1].max(dim=1).indices

single_answer_tokens = answer_tokens[single_answer_token_idxs][min_prompt_len_idxs][
    :, -1
]

correct_answer_idxs = t.where(max_tokens == single_answer_tokens)[0]

prompt_indices = t.arange(len(country_to_captial))[single_answer_token_idxs][
    min_prompt_len_idxs
][correct_answer_idxs]

valid_prompts = []
valid_answers = []
for i, (prompt, answer) in enumerate(zip(prompts, answers)):
    if i in prompt_indices:
        valid_prompts.append(prompt)
        valid_answers.append(answer)

prompt_dicts = []
for i, (valid_prompt, valid_answer) in enumerate(zip(valid_prompts, valid_answers)):
    # Choose a random index that is not i
    rand_idx = random.choice([idx for idx in range(len(valid_prompts)) if idx != i])
    prompt_dict = {
        "clean": valid_prompt,
        "corrupt": valid_prompts[rand_idx],
        "answers": [valid_answer],
        "wrong_answers": [valid_answers[rand_idx]],
    }
    prompt_dicts.append(prompt_dict)


data_json = {"prompts": prompt_dicts}

with open(f"capital_cities_{MODEL_NAME}_prompts.json", "w") as f:
    json.dump(data_json, f)
#%%
