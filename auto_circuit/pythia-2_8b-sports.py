#%%
from collections import defaultdict

import plotly.express as px
import plotly.graph_objects as go
import torch as t
import transformer_lens as tl

from auto_circuit.utils.custom_tqdm import tqdm
from auto_circuit.utils.misc import repo_path_to_abs_path

MODEL_NAME = "pythia-2.8b-deduped"
device = "cuda" if t.cuda.is_available() else "cpu"
model = tl.HookedTransformer.from_pretrained(MODEL_NAME, device=device)


def read_players(filename: str):
    filepath = repo_path_to_abs_path(f"datasets/sports-players/{filename}")
    with open(filepath, "r") as file:
        players = file.readlines()
    return [player.strip() for player in players]


football_players = read_players("american-football-players.txt")
basketball_players = read_players("basketball-players.txt")
baseball_players = read_players("baseball-players.txt")

bos = model.tokenizer.bos_token  # type: ignore
template = (
    bos + "Fact: Tiger Woods plays the sport of golf\nFact: {} plays the sport of"
)

football_prompts = [template.format(player) for player in football_players]
basketball_prompts = [template.format(player) for player in basketball_players]
baseball_prompts = [template.format(player) for player in baseball_players]

football_valid_players, basketball_valid_players, baseball_valid_players = [], [], []
ans_toks = []

for prompts, players, answer, valid_players in [
    (football_prompts, football_players, " football", football_valid_players),
    (basketball_prompts, baseball_players, " basketball", basketball_valid_players),
    (baseball_prompts, baseball_players, " baseball", baseball_valid_players),
]:
    ans_tok = model.to_tokens(answer, padding_side="left", prepend_bos=False)[0][0]
    ans_toks.append(ans_tok)
    model.tokenizer.padding_side = "left"  # type: ignore
    out = model.tokenizer(prompts, padding=True, return_tensors="pt")  # type: ignore
    prompt_tokens = out.input_ids.to(device)
    attn_mask = out.attention_mask.to(device)
    print("prompt_tokens.shape", prompt_tokens.shape)

    with t.inference_mode():
        logits = model(prompt_tokens, attention_mask=attn_mask)[:, -1]
    probs = t.softmax(logits, dim=-1)
    correct_answer_idxs = t.where(probs[:, ans_tok] > 0.5)[0]

    correct_answer_names = [players[i.item()] for i in correct_answer_idxs]
    print("correct_answer_names", correct_answer_names)
    valid_players.extend(correct_answer_names)

#%%

min_name_count = min(
    len(football_valid_players),
    len(basketball_valid_players),
    len(baseball_valid_players),
)
resids = defaultdict(list)
mean_acts = []
for layer in range(19):
    mean = []
    for sport, correct_players in [
        ("Football", football_valid_players),
        ("Basketball", basketball_valid_players),
        ("Baseball", baseball_valid_players),
    ]:
        player_prompts = [bos + name for name in correct_players]
        # template = bos + "Fact: Tiger Woods plays the sport of golf\nFact: {}"
        player_prompts = [template.format(name) for name in correct_players]
        model.tokenizer.padding_side = "left"  # type: ignore
        out = model.tokenizer(
            player_prompts, padding=True, return_tensors="pt"
        )  # type: ignore
        prompt_tokens = out.input_ids.to(device)
        attn_mask = out.attention_mask.to(device)

        with t.inference_mode():
            logits = model(
                prompt_tokens,
                attention_mask=attn_mask,
                stop_at_layer=layer + 1,  # stop_at_layer is exclusive
            )
        mean.append(logits[:min_name_count, -1].mean(dim=0))
        resids[sport].append(logits[:, -1])
    mean_acts.append(t.stack(mean).mean(dim=0))


ans_toks = t.stack(ans_toks) if isinstance(ans_toks, list) else ans_toks
#%%

ans_embeds = model.unembed.W_U[:, ans_toks]
probe = model.blocks[16].attn.W_V[20] @ model.blocks[16].attn.W_O[20] @ ans_embeds

for ans_idx, (sport, sport_resids) in enumerate(resids.items()):
    layers = list(range(2, 19))
    correct = []
    football_avgs, basketball_avgs, baseball_avgs = [], [], []
    for layer in layers:
        embed = sport_resids[layer].detach().clone()
        mean_act = mean_acts[layer].detach().clone()
        with t.inference_mode():
            # x_normed = model.blocks[layer].ln2(x).detach().clone()
            # x = x + model.blocks[layer].mlp(x_normed.unsqueeze(1)).squeeze(1)
            probe_out = (embed - mean_act) @ probe
            correct.append((probe_out.argmax(dim=-1) == ans_idx).float().mean().item())
            probs = t.nn.functional.softmax(probe_out, dim=-1)
            mean_probs = probs.mean(dim=0)
        football_avgs.append(mean_probs[0].item())
        basketball_avgs.append(mean_probs[1].item())
        baseball_avgs.append(mean_probs[2].item())

    # Plot layers vs. average probability of each sport
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=football_avgs, name="Football"))
    fig.add_trace(go.Scatter(x=layers, y=basketball_avgs, name="Basketball"))
    fig.add_trace(go.Scatter(x=layers, y=baseball_avgs, name="Baseball"))
    fig.add_trace(go.Scatter(x=layers, y=correct, name="Correct"))
    fig.update_layout(
        title="Average Probability of Each Sport",
        xaxis_title="Layer",
        yaxis_title="Probability",
    )
    fig.show()

#%%

layer_2_resids = t.cat([r[2].detach().clone() for r in resids.values()])
correct_dirs = t.cat(
    [
        probe[:, ans_idx].detach().clone().repeat(len(r[2]), 1)
        for ans_idx, r in enumerate(resids.values())
    ]
)
answer_idxs = t.cat(
    [t.tensor([ans_idx] * len(r[2])) for ans_idx, r in enumerate(resids.values())]
)
d_model = model.cfg.d_model

dataset = t.utils.data.TensorDataset(layer_2_resids, correct_dirs, answer_idxs)
train_set, test_set = t.utils.data.random_split(dataset, [0.9, 0.1])
train_loader = t.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = t.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

translate = t.zeros([d_model], device=device, requires_grad=True)
translate_2 = t.zeros([d_model], device=device, requires_grad=True)
learned_rotation = t.nn.Linear(d_model, d_model, bias=False, device=device)
linear_map = t.nn.utils.parametrizations.orthogonal(learned_rotation, "weight")
optim = t.optim.Adam(list(linear_map.parameters()) + [translate, translate_2], lr=0.01)


def pred_from_embeds(embeds: t.Tensor, lerp: float = 1.0) -> t.Tensor:
    return learned_rotation(embeds + translate) + translate_2


n_epochs = 2000
loss_history = []
for epoch in (epoch_pbar := tqdm(range(n_epochs))):
    for batch_idx, (resid, correct_dir, _) in enumerate(train_loader):
        resid = resid.to(device)
        correct_dir = correct_dir.to(device)

        optim.zero_grad()
        pred = pred_from_embeds(resid)
        loss = -t.nn.functional.cosine_similarity(pred, correct_dir).mean()
        loss_history.append(loss.item())
        loss.backward()
        optim.step()
        epoch_pbar.set_description(f"Loss: {loss.item():.3f}")

px.line(y=loss_history, title="Loss History").show()
#%%
test_corrects = []
for batch_idx, (resid, _, answer_idx) in enumerate(test_loader):
    resid = resid.to(device)
    with t.inference_mode():
        # pred = pred_from_embeds(resid)
        probe_out = resid @ probe
        pred_ans = probe_out.argmax(dim=-1)
        test_corrects.append((pred_ans == answer_idx.to(device)).float())

print("Test Accuracy:", t.cat(test_corrects).mean().item())
