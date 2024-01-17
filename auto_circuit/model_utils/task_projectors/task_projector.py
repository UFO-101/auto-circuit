from typing import Dict, Optional

import torch as t
from einops import einsum
from torch.nn.utils.parametrize import register_parametrization
from transformer_lens.hook_points import HookPoint

from auto_circuit.utils.tensor_ops import MaskFn, sample_hard_concrete

# from torch.nn.init import orthogonal


class RotationMatrix(t.nn.Module):
    def __init__(self, n_inputs: int, seq_len: Optional[int] = None) -> None:
        super().__init__()
        seq_shape = [] if seq_len is None else [seq_len]
        self.weight = t.nn.Parameter(t.empty(seq_shape + [n_inputs, n_inputs]))
        # t.nn.init.constant_(self.weight, 1.0)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x.unsqueeze(-1)
        rotated = self.weight @ x
        return rotated.squeeze(-1)

    def inverse(self, x: t.Tensor) -> t.Tensor:
        x = x.unsqueeze(-1)
        rotated = self.weight.transpose(-1, -2) @ x
        return rotated.squeeze(-1)


class Symmetric(t.nn.Module):
    def forward(self, x: t.Tensor):
        return x.triu() + x.triu(1).transpose(-1, -2)


class TaskProjector(t.nn.Module):
    """
    Task Projector

    Implements:
        latents = ReLU(encoder(x - bias) + latent_bias)
        recons = decoder(latents) + bias
    """

    def __init__(
        self,
        wrapped_hook: HookPoint,
        n_inputs: int,
        seq_len: Optional[int] = None,
        mask_fn: MaskFn = None,
    ) -> None:
        """
        :param wrapped_hook: the wrapped transformer_lens hook that caches the SAE input
        :param n_inputs: dimensionality of the input (e.g residual stream, MLP neurons)
        """
        super().__init__()
        self.wrapped_hook: HookPoint = wrapped_hook
        self.init_params(n_inputs, seq_len)
        self.mask_fn: MaskFn = mask_fn

    def init_params(self, n_inputs: int, seq_len: Optional[int] = None) -> None:
        self.n_inputs: int = n_inputs
        self.seq_len: Optional[int] = seq_len
        seq_shape = [] if seq_len is None else [seq_len]
        # rotation_matrix = RotationMatrix(n_inputs, seq_len)
        # self.rotation = orthogonal(rotation_matrix, "weight")
        # t.nn.init.orthogonal_(self.rotation.weight)
        # linear = rearrange(t.eye(n_inputs)
        eye = t.eye(n_inputs)
        eye = eye if seq_len is None else eye.unsqueeze(0).repeat(seq_len, 1, 1)
        self.linear = t.nn.Parameter(eye)
        t.nn.init.orthogonal_(self.linear)
        register_parametrization(self, "linear", Symmetric())  # type: ignore
        # self.dim_weights = t.nn.Parameter(t.zeros(seq_shape + [n_inputs]))
        # t.nn.init.constant_(self.dim_weights, 5.0)
        self.bias = t.nn.Parameter(t.zeros(seq_shape + [n_inputs]))

    def discretize_dim_weights(self, threshold: float = 0.0) -> int:
        self.dim_weights.data = (self.dim_weights.data > threshold).float()
        self.mask_fn = None
        return int(self.dim_weights.sum().item())

    @classmethod
    def from_state_dict(
        cls, wrapped_hook: HookPoint, state_dict: Dict[str, t.Tensor], mask_fn: MaskFn
    ) -> "TaskProjector":
        shape = state_dict["bias"].shape
        seq_len, n_inputs = (shape[0], shape[1]) if len(shape) > 1 else (None, shape[0])
        projector = cls(wrapped_hook, n_inputs, seq_len, mask_fn)
        projector.load_state_dict(state_dict, strict=True)
        return projector

    def encode(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: input data (shape: [..., [seq], n_inputs])
        :return: projected rotated data (shape: [..., [seq], n_inputs])
        """
        rotated = self.rotation(x)
        einstr = []
        if self.mask_fn == "hard_concrete":
            mask_weights = sample_hard_concrete(self.dim_weights, x.size(0))
            einstr.append("batch")
        elif self.mask_fn == "sigmoid":
            mask_weights = t.sigmoid(self.dim_weights)
        else:
            assert self.mask_fn is None
            mask_weights = self.dim_weights

        if self.seq_len is not None:
            einstr.append("seq")

        einstr = " ".join(einstr) + " d, batch seq d -> batch seq d"
        masked_rotated = einsum(mask_weights, rotated, einstr)
        return masked_rotated + self.bias

    def decode(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: rotated data (shape: [..., [seq], n_inputs])
        :return: unrotated data (shape: [..., [seq], n_inputs])
        """
        return self.rotation.inverse(x)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        :param x: input data (shape: [..., n_inputs])
        :return:  projected data (shape: [..., n_inputs])
        """
        # x = self.wrapped_hook(x)
        # projected_rotated = self.encode(x)
        # projected_unrotated = self.decode(projected_rotated)
        # return projected_unrotated
        transformed = (self.linear @ x.unsqueeze(-1)).squeeze(-1)
        return transformed + self.bias
