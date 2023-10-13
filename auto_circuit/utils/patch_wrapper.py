from typing import Any, Literal, Optional

import torch as t
from einops import einsum

MaskFn = Optional[Literal["hard_concrete", "sigmoid"]]

# Copied from Subnetwork Probing paper: https://github.com/stevenxcao/subnetwork-probing
left, right, temp = -0.1, 1.1, 2 / 3


class PatchWrapper(t.nn.Module):
    def __init__(
        self,
        module: t.nn.Module,
        head_dim: Optional[int] = None,
        is_src: bool = False,
        src_idxs: Optional[slice] = None,
        is_dest: bool = False,
        patch_mask: Optional[t.Tensor] = None,
        prev_src_count: Optional[int] = None,
    ):
        super().__init__()
        self.module: t.nn.Module = module
        self.head_dim: Optional[int] = head_dim
        self.curr_src_outs: Optional[t.Tensor] = None

        self.is_src = is_src
        if self.is_src:
            assert src_idxs is not None
            self.src_idxs: slice = src_idxs

        self.is_dest = is_dest
        if self.is_dest:
            assert patch_mask is not None
            self.patch_mask: t.nn.Parameter = t.nn.Parameter(patch_mask)
            self.prev_src_count: Optional[int] = prev_src_count
            self.patch_src_outs: Optional[t.Tensor] = None
            self.mask_fn: MaskFn = None
            self.dropout_layer: t.nn.Module = t.nn.Dropout(p=0.0)
        self.patch_mode = False

        self.dims = " ".join([f"d{i}" for i in range(1, head_dim)]) if head_dim else ""
        self.src_slice = slice(prev_src_count) if prev_src_count else None

    def sample_hard_concrete(self, mask: t.Tensor, batch_size: int) -> t.Tensor:
        mask = mask.repeat(batch_size, *([1] * mask.ndim))
        u = t.zeros_like(mask).uniform_().clamp(0.0001, 0.9999)
        s = t.sigmoid((u.log() - (1 - u).log() + mask) / temp)
        s_bar = s * (right - left) + left
        return s_bar.clamp(min=0.0, max=1.0)

    def sigmoid_mask(self, mask: t.Tensor) -> t.Tensor:
        return t.sigmoid(mask)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        arg_0: t.Tensor = args[0].clone()

        if self.patch_mode and self.is_dest:
            assert self.patch_src_outs is not None and self.curr_src_outs is not None
            diff = (self.patch_src_outs - self.curr_src_outs)[self.src_slice]
            batch_str, dest_str = "", "" if self.head_dim is None else "dest"
            if self.mask_fn == "hard_concrete":
                mask = self.sample_hard_concrete(self.patch_mask, arg_0.size(0))
                batch_str = "batch"
            elif self.mask_fn == "sigmoid":
                mask = self.sigmoid_mask(self.patch_mask)
            else:
                assert self.mask_fn is None
                mask = self.patch_mask
            mask = self.dropout_layer(mask)
            einsum_pre = f"{batch_str} {dest_str} src, src batch {self.dims} ..."
            einsum_post = f"batch {self.dims} {dest_str} ..."
            arg_0 += einsum(mask, diff, f"{einsum_pre} -> {einsum_post}")

        new_args = (arg_0,) + args[1:]
        out = self.module(*new_args, **kwargs)

        if self.patch_mode and self.is_src:
            assert self.curr_src_outs is not None
            if self.head_dim is None:
                src_out = out
            else:
                src_out = t.stack(out.split(1, dim=self.head_dim)).squeeze(
                    self.head_dim + 1
                )
            self.curr_src_outs[self.src_idxs] = src_out

        return out

    def __repr__(self):
        repr = [f"PatchWrapper({self.module.name})"]
        repr.append(("Src✓" if self.is_src else "") + ("Dest✓" if self.is_dest else ""))
        repr.append(f"Patch Mask: [{self.patch_mask.shape}]") if self.is_dest else None
        repr.append(str(self.patch_mask.data)) if self.is_dest else None
        return "\n".join(repr)
