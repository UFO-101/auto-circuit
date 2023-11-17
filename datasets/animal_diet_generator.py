#%%
import json
from random import choice
from typing import Any, Dict, List, Tuple

CLEAN_PROMPT_ANIMAL = "cows"  # "squirrels"  # "birds" # "mice"
ANSWER = " grass"  # " nuts"  # " worms" # " cheese"

ANIMALS = [
    "cats",
    "dogs",
    "monkeys",
    "fish",
    "horses",
    "rabbits",
    "turtles",
    "elephants",
    "lions",
    "tigers",
    "bears",
    "snakes",
    "pigs",
    "ducks",
    "chickens",
    "goats",
    "sheep",
    "frogs",
    "deer",
    "sharks",
    "whales",
]

ASKER_TUPLES: List[Tuple[str, str]] = [
    ("Tom", "his"),
    ("Jane", "her"),
    ("Lily", "her"),
    ("Sarah", "her"),
    ("Jack", "his"),
    ("Emily", "her"),
    ("Max", "his"),
    ("Oliver", "his"),
    ("Sophia", "her"),
    ("Lucas", "his"),
    ("Liam", "his"),
    ("Noah", "his"),
    ("Mia", "her"),
    ("Ethan", "his"),
    ("Grace", "her"),
    ("Daniel", "his"),
    ("Emma", "her"),
    ("Owen", "his"),
    ("Sophie", "her"),
    ("Michael", "his"),
    ("Matthew", "his"),
    ("Chloe", "her"),
    ("James", "his"),
    ("Olivia", "her"),
]

RESPONDERS = [
    "mother",
    "father",
    "teacher",
    "uncle",
    "aunt",
    "grandfather",
    "grandmother",
]

# Function to generate a single prompt with the revised askers
def generate_long_prompt(animal: str, asker: Tuple[str, str], responder: str) -> str:
    pt_1 = f'"What do {animal} like to eat?", {asker[0]} asked {asker[1]} {responder}.'
    pt_2 = f' {asker[1].capitalize()} {responder} smiled and said, "They like to eat'
    return pt_1 + pt_2


def generate_short_prompt(animal: str, asker: Tuple[str, str], responder: str) -> str:
    return f"The {animal} ate some"


# Function to generate N prompts with extended lists
def generate_prompts(N: int, short: bool = True) -> Dict[str, Any]:
    prompt_generator = generate_short_prompt if short else generate_long_prompt
    prompts = []
    for _ in range(N):
        clean_prompt = prompt_generator(
            CLEAN_PROMPT_ANIMAL, choice(ASKER_TUPLES), choice(RESPONDERS)
        )
        corrupt_prompt = prompt_generator(
            choice(ANIMALS), choice(ASKER_TUPLES), choice(RESPONDERS)
        )
        prompts.append(
            {"clean": clean_prompt, "corrupt": corrupt_prompt, "answers": [ANSWER]}
        )
    return {"prompts": prompts}


#%%
with open("animal_diet_long_prompts.json", "w") as f:
    json.dump(generate_prompts(1000, short=False), f)
#%%

# TOKENS_WITH_SPACE = [" " + word for word in ANIMALS]
# tokens = model.tokenizer(TOKENS_WITH_SPACE)['input_ids']
# [print(anim, tokens[i]) for i, anim in enumerate(TOKENS_WITH_SPACE)]
