"""Linformer backbone for batches of variable-size sets."""

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_heterogeneous_batching import Batch

from nugets.models.backbone import (
    BackBone,
    hyperparameter,
    int_hyperparameter,
    model_attribute,
)
from nugets.models.backbones.register import register


class LinformerAttention(nn.Module):
    """Multi-head self-attention with learned low-rank sequence projections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        projection_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.projection_dim = projection_dim

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.key_projection = nn.Parameter(torch.empty(max_seq_len, projection_dim))
        self.value_projection = nn.Parameter(torch.empty(max_seq_len, projection_dim))
        self.output = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.key_projection)
        nn.init.xavier_uniform_(self.value_projection)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Padded tokens must not contribute to either low-rank projection.
        node_mask = mask[:, None, :, None].to(dtype=x.dtype)
        k = k * node_mask
        v = v * node_mask

        key_projection = self.key_projection[:seq_len]
        value_projection = self.value_projection[:seq_len]
        projected_k = torch.einsum("bhnd,nr->bhrd", k, key_projection)
        projected_v = torch.einsum("bhnd,nr->bhrd", v, value_projection)

        scores = torch.matmul(q, projected_k.transpose(-2, -1)) * self.head_dim**-0.5
        attention = self.attention_dropout(scores.softmax(dim=-1))
        attended = torch.matmul(attention, projected_v)
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.output_dropout(self.output(attended))


class LinformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        projection_dim: int,
        feed_forward_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = LinformerAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, feed_forward_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


@register
class Linformer(BackBone):
    """Linformer backbone using low-rank projections along the node dimension."""

    n_heads: int = int_hyperparameter(description="number of attention heads")
    n_layers: int = int_hyperparameter(description="number of Linformer blocks")
    d_model: int = int_hyperparameter(description="input and output dimension")
    max_seq_len: int = int_hyperparameter(description="maximum number of nodes per item")
    projection_dim: int = int_hyperparameter(description="low-rank sequence dimension")
    feed_forward_hidden_dim: int = int_hyperparameter(
        description="hidden dimension of the feed-forward blocks"
    )
    dropout: float = hyperparameter(type=float, default=0.0, description="dropout")

    blocks: nn.ModuleList = model_attribute()

    def __setup__(self) -> None:
        self.blocks = nn.ModuleList(
            [
                LinformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    max_seq_len=self.max_seq_len,
                    projection_dim=self.projection_dim,
                    feed_forward_hidden_dim=self.feed_forward_hidden_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, batch, return_reg_loss=False):
        del return_reg_loss
        x_dense, mask = to_dense_batch(x=batch.data, batch=batch.batch)
        if x_dense.size(1) > self.max_seq_len:
            raise ValueError(
                f"batch contains {x_dense.size(1)} nodes, but max_seq_len="
                f"{self.max_seq_len}"
            )
        for block in self.blocks:
            x_dense = block(x_dense, mask)

        return (
            Batch.from_batched(
                data=x_dense[mask],
                order=batch.order,
                n_nodes=batch.n_nodes,
            ),
            None,
        )

    def get_input_dim(self):
        return self.d_model

    def get_output_dim(self):
        return self.d_model
