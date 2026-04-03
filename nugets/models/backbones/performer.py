import torch
import torch.nn as nn
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.utils import to_dense_batch
from torch_heterogeneous_batching import Batch

from nugets.models.backbone import BackBone, int_hyperparameter, hyperparameter, model_attribute
from nugets.models.backbones.register import register

class PerformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = PerformerAttention(
            channels=hidden_dim,
            heads=heads,
            head_channels=hidden_dim // heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x_dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_dense = x_dense + self.attn(self.norm1(x_dense), mask=mask)
        x_dense = x_dense + self.mlp(self.norm2(x_dense))
        return x_dense

@register 
class Performer(BackBone):
    """
    Performer model
    """
    n_heads: int = int_hyperparameter(description="number of heads for self-attention")
    n_layers: int = int_hyperparameter(description="number of layers")
    d_model: int = int_hyperparameter(description="number of dimensions of the input and output")
    dropout: float = hyperparameter(default=0.0, type=float, description="dropout")

    def __setup__(self):
        self.blocks = nn.ModuleList([PerformerBlock(self.d_model, self.n_heads, self.dropout) for _ in range(self.n_layers)])

    def forward(self, batch, return_reg_loss=False):
        x_dense, mask = to_dense_batch(x=batch.data, batch=batch.batch)
        for block in self.blocks:
            x_dense = block(x_dense, mask)
        x_nodes = x_dense[mask]
        return Batch.from_batched(data = x_nodes, order=batch.order, n_nodes = batch.n_nodes, ), None
    
    def get_input_dim(self):
        return self.d_model
    def get_output_dim(self):
        return self.d_model