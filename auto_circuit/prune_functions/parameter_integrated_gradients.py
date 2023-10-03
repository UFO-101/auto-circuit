# class BaselineWeights(Enum):
#     """The baseline weights to use for integrated gradients."""

#     ZERO = 0
#     MEAN = 1
#     PERMUTED = 2


# def parameter_integrated_grads_prune_scores(
#     model: t.nn.Module,
#     factorized: bool,
#     train_data: DataLoader[PromptPairBatch],
#     baseline_weights: BaselineWeights,
#     samples: int = 50,
# ) -> Dict[Edge, float]:
#     """Gradients of weights wrt to network output, integrated between some baseline
#     weights and the actual weights."""

#     edges: OrderedSet[Edge] = graph_edges(model, factorized)
#     weights = set(chain(*[[e.src.weight, e.dest.weight] for e in edges]))
#     weights = list(filter(lambda x: x is not None, weights))

#     normal_state = {}
#     for name, param in model.state_dict().items():
#         if name in weights:
#             assert isinstance(param, t.Tensor)
#             normal_state[name] = param.clone()

#     base_state = {}
#     if baseline_weights == BaselineWeights.ZERO:
#         base_state = dict([(n, t.zeros_like(p)) for n, p in normal_state.items()])
#     else:
#         raise NotImplementedError

#     ig = dict([(n, t.zeros_like(p)) for n, p in normal_state.items()])
#     for idx in (pbar := tqdm(range(samples))):
#         pbar.set_description_str(f"PIG Sample {idx+1}/{samples}", refresh=False)
#         lerp_state = {}
#         for name in weights:
#             lerp_state[name] = base_state[name] + (
#                 normal_state[name] - base_state[name]
#             ) * idx / (samples - 1)
#             model.load_state_dict(lerp_state, strict=False)

#         model.zero_grad()
#         for batch in train_data:
#             out = model(batch.clean)
#             loss = out.mean()
#             loss.backward()

#         for name, param in model.named_parameters():
#             if name in weights:
#                 assert isinstance(param.grad, t.Tensor)
#                 ig[name] += param.grad / samples

#     weight_diffs = {}
#     for name, normal_param in normal_state.items():
#         weight_diffs[name] = normal_param - base_state[name]

#     for name, val in ig.items():
#         ig[name] = val * weight_diffs[name]

#     prune_scores = {}
#     for edge in edges:
#         if factorized:
#             src_ig = ig[edge.src.weight][edge.src.weight_t_idx]
#             dest_ig = ig[edge.dest.weight][edge.dest.weight_t_idx]
#             prune_scores[edge] = (
#                 src_ig.abs().sum().item() + dest_ig.abs().sum().item()
#             ) / 2
#         else:
#             src_ig = ig[edge.src.weight][edge.src.weight_t_idx]
#             prune_scores[edge] = src_ig.abs().sum().item()

#     return prune_scores
