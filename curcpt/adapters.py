from __future__ import annotations

import torch
import torch.nn as nn


class HeadSliceLoRA(nn.Module):
    """LoRA adapter for a single Q head's slice of q_proj or o_proj.

    Operates on head_dim slice, not full projection.
    Standard LoRA init: A is random and B is zero, so the adapter starts as a
    no-op but still has a live gradient path. Zero-initializing both A and B
    makes the bilinear adapter permanently dead.
    """

    def __init__(self, head_dim: int, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(rank, head_dim))
        self.lora_B = nn.Parameter(torch.zeros(head_dim, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x_slice: torch.Tensor) -> torch.Tensor:
        weight_dtype = self.lora_A.dtype
        out = (x_slice.to(weight_dtype) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out.to(x_slice.dtype)


class HeadAdapterSet(nn.Module):
    """Container for all head-specific LoRA adapters across retrieval heads.

    Each retrieval head (layer_id, q_head_id) gets:
      - q_lora: HeadSliceLoRA for q_proj slice
      - o_lora: HeadSliceLoRA for o_proj slice
    """

    def __init__(
        self,
        retrieval_heads: set[tuple[int, int]],
        head_dim: int,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.retrieval_heads = frozenset(retrieval_heads)
        self.head_dim = head_dim
        self.rank = rank
        self.alpha = alpha

        self.q_loras = nn.ModuleDict()
        self.o_loras = nn.ModuleDict()

        for layer_id, q_head_id in sorted(self.retrieval_heads):
            key = f"{layer_id}_{q_head_id}"
            self.q_loras[key] = HeadSliceLoRA(head_dim, rank, alpha)
            self.o_loras[key] = HeadSliceLoRA(head_dim, rank, alpha)

    def has_adapter(self, layer_id: int, q_head_id: int) -> bool:
        return (layer_id, q_head_id) in self.retrieval_heads

    def get_q_lora(self, layer_id: int, q_head_id: int) -> HeadSliceLoRA | None:
        key = f"{layer_id}_{q_head_id}"
        if key in self.q_loras:
            return self.q_loras[key]
        return None

    def get_o_lora(self, layer_id: int, q_head_id: int) -> HeadSliceLoRA | None:
        key = f"{layer_id}_{q_head_id}"
        if key in self.o_loras:
            return self.o_loras[key]
        return None

    def adapter_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self.q_loras.values():
            params.extend(module.parameters())
        for module in self.o_loras.values():
            params.extend(module.parameters())
        return params

    def num_heads(self) -> int:
        return len(self.retrieval_heads)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.adapter_parameters())
