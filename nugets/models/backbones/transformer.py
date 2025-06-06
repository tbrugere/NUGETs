from torch_heterogeneous_batching.nn.transformer import Transformer as Transformer_nn
from torch_geometric.nn.resolver import aggregation_resolver

from nugets.models.backbone import BackBone, int_hyperparameter, hyperparameter, model_attribute
from nugets.models.backbones.register import register

@register
class Transformer(BackBone):
    """Transformer backbone, vanilla transformer without any positional encodings"""

    n_heads: int = int_hyperparameter(description="number of heads for the self-attentions")
    n_layers: int = int_hyperparameter(description="number of layers")
    d_model: int = int_hyperparameter(description="number of dimensions of the input and the output")
    
    # for simplicity, key_dim is always set to d_model / n_heads (to have key = query = value )
    # key_dim: int = int_hyperparameter(description="number of dimensions for key, query"
    #                              " and values in attention mechanism")
    feed_forward_hidden_dim: int = int_hyperparameter(description="number of hidden dimensions"
                                 " in feed-forward blocks")


    aggregation: str = hyperparameter(type=str, description="sequence aggregation function")
    transformer: Transformer_nn = model_attribute()

    def __setup__(self):
        self.transformer = Transformer_nn(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            input_dim=self.d_model,
            key_dim=None,
            hidden_dim=self.feed_forward_hidden_dim,
        )
        if self.aggregation != "none":
            aggregation_args = {}
            self.aggregation_fn = aggregation_resolver(self.aggregation, **aggregation_args)

    def forward(self, batch, return_reg_loss=False):
        del return_reg_loss # no regularization loss for transformer
        if self.aggregation != "none":
            return self.aggregation_fn(self.transformer(batch).data, ptr=batch.ptr), None
        return self.transformer(batch), None

    def get_input_dim(self): return self.d_model
    def get_output_dim(self): return self.d_model


