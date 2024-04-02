# WikiData SPARQL query:
"""
SELECT ?itemLabel WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
  {
    FILTER(EXISTS {
      ?article schema:about ?item;
        schema:isPartOf <https://en.wikipedia.org/>.
    })
    {
      SELECT ?item (COUNT(DISTINCT ?article) AS ?n_wikipedia_articles)
                                    (COUNT(DISTINCT ?occs) AS ?n_occupations) WHERE {
        ?item wdt:P106 wd:Q10871364.
        ?article schema:about ?item.
        ?item wdt:P106 ?occs.
      }
      GROUP BY ?item
      ORDER BY DESC (?count)
      LIMIT 2194
    }
  }
}
"""
#%%
import json
import random
from typing import Dict, List

import torch as t
import transformer_lens as tl

from auto_circuit.utils.misc import repo_path_to_abs_path


#%%
def read_players(filename: str):
    filepath = repo_path_to_abs_path(f"datasets/sports-players/{filename}")
    with open(filepath, "r") as file:
        players = file.readlines()
    return [player.strip() for player in players]


sport_players: Dict[str, List[str]] = {
    "football": read_players("american-football-players.txt"),
    "basketball": read_players("basketball-players.txt"),
    "baseball": read_players("baseball-players.txt"),
}
template = "Fact: Tiger Woods plays the sport of golf\nFact: {} plays the sport of"

sport_prompts: Dict[str, List[str]] = {
    "football": [template.format(player) for player in sport_players["football"]],
    "basketball": [template.format(player) for player in sport_players["basketball"]],
    "baseball": [template.format(player) for player in sport_players["baseball"]],
}

MODEL_NAME = "pythia-2.8b-deduped"

device = "cuda" if t.cuda.is_available() else "cpu"
model = tl.HookedTransformer.from_pretrained(MODEL_NAME, device=device)

#%%
name_length = 3
correct_prompt_len = 16 + name_length


def valid_idxs(sport: str, prompts: List[str]) -> List[int]:
    ans_str = " " + sport
    ans_tok = model.to_tokens(ans_str, padding_side="left", prepend_bos=False)[0][0]
    prompt_tokens = model.to_tokens(prompts, prepend_bos=False, padding_side="left")
    print("prompt_tokens.shape", prompt_tokens.shape)

    padding_token = model.tokenizer.pad_token_id  # type: ignore
    correct_prompt_len_idxs = t.where(
        # <pad> == <bos> so we need to subtract 1 from the correct prompt len
        (prompt_tokens != padding_token).sum(dim=1)
        == correct_prompt_len
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
    final_idxs = final_idxs.tolist()
    random.shuffle(final_idxs)
    return final_idxs


sport_valid_idxs = dict([(k, valid_idxs(k, v)) for k, v in sport_prompts.items()])

min_num_players: int = min([len(sport_valid_idxs[k]) for k in sport_prompts.keys()])

player_dicts = []
prompt_dicts = []
for sport, idxs in sport_valid_idxs.items():
    incorrect_players = [s for s in sport_players.keys() if s != sport]
    incorrect_answers = [s for s in sport_prompts.keys() if s != sport]
    clean_players = sport_players[sport]
    clean_prompts = sport_prompts[sport]
    for i in range(min_num_players):
        # Randomly choose a different sport
        rand_incorrect_sport = random.choice(incorrect_answers)
        corrupt_players = sport_players[rand_incorrect_sport]
        corrupt_prompts = sport_prompts[rand_incorrect_sport]
        corrupt_valid_idx = random.choice(sport_valid_idxs[rand_incorrect_sport])
        player_dict = {
            "clean": " " + clean_players[idxs[i]],
            "corrupt": " " + corrupt_players[corrupt_valid_idx],
            "answers": [" " + sport],
            "wrong_answers": [" " + s for s in incorrect_players],
        }
        prompt_dict = {
            "clean": clean_prompts[idxs[i]],
            "corrupt": corrupt_prompts[corrupt_valid_idx],
            "answers": [" " + sport],
            "wrong_answers": [" " + s for s in incorrect_answers],
        }
        player_dicts.append(player_dict)
        prompt_dicts.append(prompt_dict)

player_data_json = {
    "word_idxs": {
        "first_name_tok": 0,
        "final_name_tok": name_length - 1,
        "end": name_length - 1,
    },
    "prompts": player_dicts,
}
prompt_data_json = {
    "word_idxs": {
        "first_name_tok": 12,
        "final_name_tok": 12 + name_length - 1,
        "end": correct_prompt_len - 1,
    },
    "prompts": prompt_dicts,
}

#%%
repo_path = f"datasets/sports-players/sports_players_{MODEL_NAME}_names.json"
with open(repo_path_to_abs_path(repo_path), "w") as f:
    json.dump(player_data_json, f)

repo_path = f"datasets/sports-players/new_sports_players_{MODEL_NAME}_prompts.json"
with open(repo_path_to_abs_path(repo_path), "w") as f:
    json.dump(prompt_data_json, f)
#%%
