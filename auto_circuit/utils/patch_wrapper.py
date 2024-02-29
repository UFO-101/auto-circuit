from typing import Any, Optional

import torch as t
from einops import einsum

from auto_circuit.types import MaskFn, PatchWrapper
from auto_circuit.utils.tensor_ops import sample_hard_concrete


class PatchWrapperImpl(PatchWrapper):
    def __init__(
        self,
        module_name: str,
        module: t.nn.Module,
        head_dim: Optional[int] = None,
        seq_dim: Optional[int] = None,
        is_src: bool = False,
        src_idxs: Optional[slice] = None,
        is_dest: bool = False,
        patch_mask: Optional[t.Tensor] = None,
        prev_src_count: Optional[int] = None,
    ):
        super().__init__()
        self.module_name: str = module_name
        self.module: t.nn.Module = module
        self.head_dim: Optional[int] = head_dim
        self.seq_dim: Optional[int] = seq_dim
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

        assert head_dim is None or seq_dim is None or head_dim > seq_dim
        dims = range(1, max(head_dim if head_dim else 2, seq_dim if seq_dim else 2))
        self.dims = " ".join(["seq" if i == seq_dim else f"d{i}" for i in dims])
        self.src_slice = slice(prev_src_count) if prev_src_count else None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        arg_0: t.Tensor = args[0].clone()

        if self.patch_mode and self.is_dest:
            assert self.patch_src_outs is not None and self.curr_src_outs is not None
            d = self.patch_src_outs[self.src_slice] - self.curr_src_outs[self.src_slice]
            batch_str = ""
            head_str = "" if self.head_dim is None else "dest"  # Patch heads separately
            seq_str = "" if self.seq_dim is None else "seq"  # Patch tokens separately
            if self.mask_fn == "hard_concrete":
                mask = sample_hard_concrete(self.patch_mask, arg_0.size(0))
                batch_str = "batch"  # Sample distribution for each batch element
            elif self.mask_fn == "sigmoid":
                mask = t.sigmoid(self.patch_mask)
            else:
                assert self.mask_fn is None
                mask = self.patch_mask
            mask = self.dropout_layer(mask)
            ein_pre = f"{batch_str} {seq_str} {head_str} src, src batch {self.dims} ..."
            ein_post = f"batch {self.dims} {head_str} ..."
            arg_0 += einsum(mask, d, f"{ein_pre} -> {ein_post}")  # Add mask times diff

        new_args = (arg_0,) + args[1:]
        out = self.module(*new_args, **kwargs)

        if self.patch_mode and self.is_src:
            assert self.curr_src_outs is not None
            if self.head_dim is None:
                src_out = out
            else:
                squeeze_dim = self.head_dim if self.head_dim < 0 else self.head_dim + 1
                src_out = t.stack(out.split(1, dim=self.head_dim)).squeeze(squeeze_dim)
            self.curr_src_outs[self.src_idxs] = src_out

        return out

    def __repr__(self):
        module_str = self.module.name if hasattr(self.module, "name") else self.module
        repr = [f"PatchWrapper({module_str})"]
        repr.append(("Src✓" if self.is_src else "") + ("Dest✓" if self.is_dest else ""))
        repr.append(f"Patch Mask: [{self.patch_mask.shape}]") if self.is_dest else None
        repr.append(str(self.patch_mask.data)) if self.is_dest else None
        return "\n".join(repr)
