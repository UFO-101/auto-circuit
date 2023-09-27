import math
from functools import partial
from typing import Any, Dict, List, Set, Tuple

import einops
import torch as t
import transformer_lens
from ordered_set import OrderedSet

import auto_circuit.model_utils.micro_model_utils as mm_utils
import auto_circuit.model_utils.transformer_lens_utils as tl_utils
from auto_circuit.model_utils.micro_model_utils import MicroModel
from auto_circuit.types import (
    DestNode,
    Edge,
    EdgeCounts,
    SrcNode,
    TensorIndex,
    TestEdges,
)
from auto_circuit.utils.misc import remove_hooks


def graph_edges(
    model: t.nn.Module, factorized: bool, reverse_topo_sort: bool = False
) -> OrderedSet[Edge]:
    """Get the edges of the computation graph of the model."""
    if not factorized:
        if isinstance(model, MicroModel):
            edge_set: OrderedSet[Edge] = mm_utils.simple_graph_edges(model)
        elif isinstance(model, transformer_lens.HookedTransformer):
            edge_set: OrderedSet[Edge] = tl_utils.simple_graph_edges(model)
        else:
            raise NotImplementedError(model)

        if reverse_topo_sort:
            edge_set = OrderedSet([edge for edge in edge_set][::-1])
        return edge_set
    else:
        if isinstance(model, MicroModel):
            src_lyrs: List[OrderedSet[SrcNode]] = mm_utils.fctrzd_graph_src_lyrs(model)
            dest_lyrs: List[OrderedSet[DestNode]] = mm_utils.fctrzd_graph_dest_lyrs(
                model
            )
        elif isinstance(model, transformer_lens.HookedTransformer):
            src_lyrs: List[OrderedSet[SrcNode]] = tl_utils.fctrzd_graph_src_lyrs(model)
            dest_lyrs: List[OrderedSet[DestNode]] = tl_utils.fctrzd_graph_dest_lyrs(
                model
            )
        else:
            raise NotImplementedError(model)

        edges = []
        if reverse_topo_sort is False:
            for src_layer, layer_srcs in enumerate(src_lyrs):
                for src in layer_srcs:
                    for layer_dests in dest_lyrs[src_layer:]:
                        for dest in layer_dests:
                            edges.append(Edge(src=src, dest=dest))
        else:
            for dest_layer, layer_dests in list(enumerate(dest_lyrs))[::-1]:
                for dest in layer_dests[::-1]:
                    for layer_srcs in src_lyrs[dest_layer::-1]:
                        for src in layer_srcs[::-1]:
                            edges.append(Edge(dest=dest, src=src))
        return OrderedSet(edges)


def graph_src_nodes(model: t.nn.Module, factorized: bool) -> Set[SrcNode]:
    """Get the src nodes of the computational graph of the model."""
    edges = graph_edges(model, factorized)
    return set([edge.src for edge in edges])


def graph_dest_nodes(model: t.nn.Module, factorized: bool) -> Set[DestNode]:
    """Get the dest nodes of the computational graph of the model."""
    edges = graph_edges(model, factorized)
    return set([edge.dest for edge in edges])


def edge_counts_util(
    model: t.nn.Module,
    factorized: bool,
    test_counts: TestEdges,
) -> List[int]:
    edges = graph_edges(model, factorized)
    n_edges = len(edges)

    if test_counts == EdgeCounts.ALL:
        counts_list = [n for n in range(n_edges + 1)]
    elif test_counts == EdgeCounts.LOGARITHMIC:
        counts_list = [
            n
            for n in range(n_edges + 1)
            if n % 10 ** math.floor(math.log10(max(n, 1))) == 0
        ]
    elif isinstance(test_counts, List):
        counts_list = [n if type(n) == int else int(n_edges * n) for n in test_counts]
    else:
        raise NotImplementedError(f"Unknown test_counts: {test_counts}")

    return counts_list


def src_out_hook(
    model: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    edge_src: SrcNode,
    src_outs: Dict[SrcNode, t.Tensor],
):
    src_outs[edge_src] = output[edge_src.out_idx]


def get_src_outs(
    model: t.nn.Module, nodes: Set[SrcNode], input: t.Tensor
) -> Dict[SrcNode, t.Tensor]:
    node_outs: Dict[SrcNode, t.Tensor] = {}
    with remove_hooks() as handles:
        for node in nodes:
            hook_fn = partial(src_out_hook, edge_src=node, src_outs=node_outs)
            handles.add(node.module(model).register_forward_hook(hook_fn))
        with t.inference_mode():
            model(input)
    return node_outs


# class PatchInput(t.nn.Module):
#     def __init__(
#         self,
#         module: t.nn.Module,
#         patches: Dict[Edge, t.Tensor],
#         src_outs: Dict[SrcNode, t.Tensor],
#     ):
#         super().__init__()
#         self.module: t.nn.Module = module
#         self.patches: Dict[Edge, t.Tensor] = patches
#         self.src_outs: Dict[SrcNode, t.Tensor] = src_outs

#     def forward(self, *args: Any, **kwargs: Any) -> t.Tensor:
#         arg_0 = args[0].clone().detach()
#         for edge, patch in self.patches.items():
#             arg_0[edge.dest.in_idx] = (
#                 arg_0[edge.dest.in_idx] + patch - self.src_outs[edge.src]
#             )

#         new_args = (arg_0, *args[1:])
#         return self.module(*new_args, **kwargs)

CombinedIndex = Tuple[slice | List[int], ...] | slice | List[int]


def get_head_dim(dest: DestNode) -> int | None:
    if type(dest.in_idx) == slice:
        assert dest.in_idx == slice(None)
        return None
    elif type(dest.in_idx) == int:
        return 0
    else:
        assert type(dest.in_idx) == tuple
        for i, idx in enumerate(dest.in_idx):
            if type(idx) == slice:
                assert idx == slice(None)
            else:
                assert type(idx) == int
                return i


def tensor_idx_to_combined_idx(t_idx: TensorIndex) -> CombinedIndex:
    if type(t_idx) == int:
        return [t_idx]
    elif type(t_idx) == slice:
        assert t_idx == slice(None)
        return t_idx
    else:
        assert type(t_idx) == tuple
        combined = []
        for idx in tuple(t_idx):
            new_idx = tensor_idx_to_combined_idx(idx)
            combined.append(new_idx)
        return tuple(combined)


def add_tensor_idx_to_combined_idx(
    t_idx: TensorIndex, combined_idx: CombinedIndex
) -> CombinedIndex:
    if type(combined_idx) == List[int]:
        assert type(t_idx) == int
        combined_idx.append(t_idx)
    elif type(combined_idx) == slice:
        assert t_idx == slice(None)
    else:
        assert type(combined_idx) == tuple and type(t_idx) == tuple
        for c_idx, idx in zip(combined_idx, t_idx):
            if type(c_idx) == slice:
                assert c_idx == slice(None) and idx == slice(None)
            else:
                c_idx.append(idx)  # type: ignore
    return combined_idx


TotalIndex = Tuple[slice | int | List[int], ...] | slice | int | List[int]


def combine_non_tuple(c_idx: CombinedIndex, patch_slice: TensorIndex) -> TotalIndex:
    assert type(c_idx) != tuple and type(patch_slice) != tuple
    if c_idx == slice(None):
        return patch_slice
    else:
        assert type(c_idx) == slice
        return c_idx


def patch_idx_and_patch_slice(c_idx: CombinedIndex, patch_slice: TensorIndex) -> TotalIndex:
    if type(c_idx) != tuple and type(patch_slice) != tuple:
        return combine_non_tuple(c_idx, patch_slice)
    elif type(c_idx) == tuple and type(patch_slice) != tuple:
        first_elem = combine_non_tuple(c_idx[0], patch_slice)
        return (first_elem, *c_idx[1:])  # type: ignore
    elif type(c_idx) != tuple and type(patch_slice) == tuple:
        first_elem = combine_non_tuple(c_idx, patch_slice[0])
        return (first_elem, *patch_slice[1:])  # type: ignore
    else:
        assert type(c_idx) == tuple and type(patch_slice) == tuple
        new_idx = []
        for i in range(max(len(c_idx), len(patch_slice))):
            if i > len(c_idx) - 1:
                new_idx.append(patch_slice[i])
            elif i > len(patch_slice) - 1:
                new_idx.append(c_idx[i])
            else:
                new_idx.append(patch_idx_and_patch_slice(c_idx[i], patch_slice[i]))
        return tuple(new_idx)


class PatchInput(t.nn.Module):
    def __init__(
        self,
        module: t.nn.Module,
        # srcs_to_patch: Dict[HashableTensorIndex, t.Tensor],  # Dict[head_idx, [src]]
        patch_idx: CombinedIndex,
        srcs_to_patch: t.Tensor,  # [src] [dest_heads, src]
        src_outs: t.Tensor,  # [src, batch, resid] [src, batch, tok, resid]
        patch_outs: t.Tensor,  # [src, batch, resid] [src, batch, tok, resid]
        head_dim: int | None,
        patch_slice: TensorIndex,
    ):
        super().__init__()
        self.module: t.nn.Module = module
        # self.srcs_to_patch: Dict[HashableTensorIndex, t.Tensor] = srcs_to_patch
        self.patch_idx: CombinedIndex = patch_idx
        self.srcs_to_patch: t.Tensor = srcs_to_patch
        self.src_outs: t.Tensor = src_outs  # Gets update by src_out_hook
        self.patch_outs: t.Tensor = patch_outs  # Fixed
        self.head_dim: int | None = head_dim
        self.patch_slice: TensorIndex = patch_slice
        # if self.patch_slice != slice(None) and self.head_dim is not None:
        #     if type(self.patch_slice == int):
        #         self.head_dim -= 1
        #     else:
        #         assert type(self.patch_slice) == tuple
        #         if len(self.patch_slice) < self.head_dim:
        #             self.head_dim -= 1

    def forward(self, *args: Any, **kwargs: Any) -> Any:

        # for patch_idx, patch in self.srcs_to_patch.items():
        #     arg_0[tensor_index_to_slice(patch_idx)][self.output_idx] += einops.einsum(
        #         patch,
        #         self.patch_outs - self.src_outs,
        #         "src, src batch resid -> batch resid",
        #     )
        arg_0 = args[
            0
        ].clone()  # [batch, resid] [batch, tok, resid] [batch, all_heads, resid] [batch, tok, all_heads, resid]
        total_idx = patch_idx_and_patch_slice(self.patch_idx, self.patch_slice)

        diff = (
            self.patch_outs - self.src_outs
        )  # [src, batch, resid] [src, batch, tok, resid]
        if self.head_dim is None:
            assert self.patch_idx == slice(None)
            arg_0[self.patch_slice] += einops.einsum(
                self.srcs_to_patch, diff, "src, src ... -> ..."
            )
        else:
            assert self.patch_idx != slice(None)
            if self.head_dim == 0:
                assert type(self.patch_idx) == List
                arg_0[total_idx] += einops.einsum(
                    self.srcs_to_patch, diff, "dest_idx src, src ... -> dest_idx ..."
                )
            elif self.head_dim == 1:
                assert type(self.patch_idx) == tuple
                arg_0[total_idx] += einops.einsum(
                    self.srcs_to_patch,
                    diff,
                    "dest_idx src, src dim1 ... -> dim1 dest_idx ...",
                )
            elif self.head_dim == 2:
                assert type(self.patch_idx) == tuple
                arg_0[total_idx] += einops.einsum(
                    self.srcs_to_patch,
                    diff,
                    "dest_idx src, src dim1 dim2 ... -> dim1 dim2 dest_idx ...",
                )
            else:
                raise NotImplementedError("head_dim > 2 not implemented")
            # args[0][total_idx] = arg_0

        new_args = (arg_0, *args[1:])
        return self.module(*new_args, **kwargs)
        # return self.module(*args, **kwargs)


def input_hook(
    module: t.nn.Module,
    input: Tuple[t.Tensor, ...],
    output: t.Tensor,
    node: DestNode,
    input_dict: Dict[DestNode, t.Tensor],
) -> None:
    if isinstance(module, PatchInput):
        input_dict[node] = output[node.in_idx]
    else:
        input_dict[node] = input[0][node.in_idx]


def get_dest_ins(
    model: t.nn.Module, nodes: Set[DestNode], input: t.Tensor
) -> Dict[DestNode, t.Tensor]:
    node_inputs: Dict[DestNode, t.Tensor] = {}
    with remove_hooks() as handles:
        for node in nodes:
            hook_fn = partial(input_hook, node=node, input_dict=node_inputs)
            handles.add(node.module(model).register_forward_hook(hook_fn))
        with t.inference_mode():
            model(input)
    return node_inputs
