#%%
import json
from random import choice
from typing import Any, Dict, List, Tuple

import transformer_lens as tl

CLEAN_PROMPT_ANIMAL = "cows"  # "squirrels"  # "birds" # "mice"
ANSWER = " grass"  # " nuts"  # " worms" # " cheese"

ANIMALS = {
    "cats": "mice",
    "dogs": "bones",
    "monkeys": "bananas",
    "fish": "worms",
    "horses": "hay",
    "rabbits": "carrots",
    "turtles": "lettuce",
    "elephants": "peanuts",
    "lions": "prey",
    "tigers": "prey",
    "bears": "fish",
    "snakes": "mice",
    "pigs": "corn",
    "ducks": "bread",
    "chickens": "corn",
    # "goats": "grass",
    # "sheep": "grass",
    "frogs": "flies",
    # "deer": "grass",
    "sharks": "fish",
    "whales": "fish",
}

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
        corrupt_animal, corrupt_answer = choice(list(ANIMALS.items()))
        corrupt_prompt = prompt_generator(
            corrupt_animal, choice(ASKER_TUPLES), choice(RESPONDERS)
        )
        prompts.append(
            {
                "clean": clean_prompt,
                "corrupt": corrupt_prompt,
                "answers": [ANSWER],
                "wrong_answers": [" " + corrupt_answer],
            }
        )
    return {"prompts": prompts}


#%%
with open("animal_diet_short_prompts.json", "w") as f:
    json.dump(generate_prompts(1000, short=True), f)
#%%
model = tl.HookedTransformer.from_pretrained("gpt2")
ANIMALS_WITH_SPACE = [" " + animal for animal in ANIMALS.keys()]
WRONG_ANSWERS_WITH_SPACE = [" " + animal for animal in ANIMALS.values()]
for word_list in [ANIMALS_WITH_SPACE, WRONG_ANSWERS_WITH_SPACE]:
    tokens = model.tokenizer(word_list)["input_ids"]  # type: ignore
    [print(anim, tokens[i]) for i, anim in enumerate(word_list)]  # type: ignore
