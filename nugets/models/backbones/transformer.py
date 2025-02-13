from torch_heterogeneous_batching.nn.transformer import Transformer as Transformer_nn

from nugets.models.backbone import BackBone, int_hyperparameter, model_attribute
from nugets.models.backbones.register import register

@register
class Transformer(BackBone):
    """Transformer backbone"""

    n_heads: int = int_hyperparameter(description="number of heads for the self-attentions")
    n_layers: int = int_hyperparameter(description="number of layers")
    d_model: int = int_hyperparameter(description="number of dimensions of the input and the output")
    
    key_dim: int = int_hyperparameter(description="number of dimensions for key, query"
                                 " and values in attention mechanism")
    feed_forward_hidden_dim: int = int_hyperparameter(description="number of hidden dimensions"
                                 " in feed-forward blocks")

    transformer: Transformer_nn = model_attribute()

    def __setup__(self):
        self.transformer = Transformer_nn(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            input_dim=self.d_model,
            key_dim=self.key_dim,
            hidden_dim=self.feed_forward_hidden_dim,
        )

    def forward(self, batch, return_reg_loss=False):
        del return_reg_loss # no regularization loss for transformer
        return self.transformer(batch), None

    def get_input_dim(self): return self.d_model
    def get_output_dim(self): return self.d_model


