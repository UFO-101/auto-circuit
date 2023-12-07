#%%
import json

import torch as t
import transformer_lens as tl

NOUNS = [
    "abduction",
    "accord",
    "affair",
    "agreement",
    "appraisal",
    "assaults",
    "assessment",
    "attack",
    "attempts",
    "campaign",
    "captivity",
    "case",
    "challenge",
    "chaos",
    "clash",
    "collaboration",
    "coma",
    "competition",
    "confrontation",
    "consequence",
    "conspiracy",
    "construction",
    "consultation",
    "contact",
    "contract",
    "convention",
    "cooperation",
    "custody",
    "deal",
    "decline",
    "decrease",
    "demonstrations",
    "development",
    "disagreement",
    "disorder",
    "dispute",
    "domination",
    "dynasty",
    "effect",
    "effort",
    "employment",
    "endeavor",
    "engagement",
    "epidemic",
    "evaluation",
    "exchange",
    "existence",
    "expansion",
    "expedition",
    "experiments",
    "fall",
    "fame",
    "flights",
    "friendship",
    "growth",
    "hardship",
    "hostility",
    "illness",
    "impact",
    "imprisonment",
    "improvement",
    "incarceration",
    "increase",
    "insurgency",
    "invasion",
    "investigation",
    "journey",
    "kingdom",
    "marriage",
    "modernization",
    "negotiation",
    "notoriety",
    "obstruction",
    "operation",
    "order",
    "outbreak",
    "outcome",
    "overhaul",
    "patrols",
    "pilgrimage",
    "plague",
    "plan",
    "practice",
    "process",
    "program",
    "progress",
    "project",
    "pursuit",
    "quest",
    "raids",
    "reforms",
    "reign",
    "relationship",
    "retaliation",
    "riot",
    "rise",
    "rivalry",
    "romance",
    "rule",
    "sanctions",
    "shift",
    "siege",
    "slump",
    "statute",  # Changed from stature, which I think is a mistake
    "stint",
    "strikes",
    "study",
    "test",
    "testing",
    "tests",
    "therapy",
    "tour",
    "tradition",
    "treaty",
    "trial",
    "trip",
    "unemployment",
    "voyage",
    "warfare",
    "work",
]


class GreaterThanConstants:
    YEARS: list[str]
    YEARS_BY_CENTURY: dict[str, list[str]]

    def __init__(self, model):
        _TOKENIZER = model.tokenizer
        del model

        self.YEARS = []
        self.YEARS_BY_CENTURY = {}

        for century in range(11, 18):
            all_success = []
            for year in range(century * 100 + 2, (century * 100) + 99):
                a = _TOKENIZER.encode(f" {year}")
                if a == [
                    _TOKENIZER.encode(f" {str(year)[:2]}")[0],
                    _TOKENIZER.encode(str(year)[2:])[0],
                ]:
                    # Ensure XX >> 01 in YYXX so the 01 version lowers the answer logits
                    if int(str(year)[2:]) > 40:
                        all_success.append(str(year))
                    continue
            self.YEARS.extend(all_success[1:-1])
            self.YEARS_BY_CENTURY[century] = all_success[1:-1]


def get_year_data(num_examples, model):
    _TOKENIZER = model.tokenizer
    constants = GreaterThanConstants(model)

    template = "The {noun} lasted from the year {year1} to "

    # set some random seed
    t.random.manual_seed(54)
    nouns_perm = t.randint(0, len(NOUNS), (num_examples,))
    years_perm = t.randint(0, len(constants.YEARS), (num_examples,))

    prompt_dicts = []
    for i in range(num_examples):
        yr = constants.YEARS[years_perm[i]]
        clean = template.format(noun=NOUNS[nouns_perm[i]], year1=yr) + yr[:2]
        century = int(yr[:2])
        zero_one_year = str(century) + "01"
        combined_toks = _TOKENIZER.encode(f" {zero_one_year}")
        separate_toks = [
            _TOKENIZER.encode(f" {zero_one_year[:2]}")[0],
            _TOKENIZER.encode(zero_one_year[2:])[0],
        ]
        assert combined_toks == separate_toks
        corrupt = (
            template.format(noun=NOUNS[nouns_perm[i]], year1=zero_one_year) + yr[:2]
        )
        answers = [str(y) for y in range(int(yr[2:]) + 1, 100)]
        wrong_answers = [
            ("0" if y <= 9 else "") + str(y) for y in range(0, int(yr[2:]) + 1)
        ]
        prompt_dict = {
            "clean": clean,
            "corrupt": corrupt,
            "answers": answers,
            "wrong_answers": wrong_answers,
        }
        prompt_dicts.append(prompt_dict)
    return prompt_dicts


model_name = "gpt2-small"
model = tl.HookedTransformer.from_pretrained("gpt2-small")
prompt_dicts = get_year_data(1000, model)
data_json = {"prompts": prompt_dicts}

with open(f"greaterthan_{model_name}_prompts.json", "w") as f:
    json.dump(data_json, f)
