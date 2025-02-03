from torch_heterogeneous_batching.nn.transformer import Transformer_nn

from nugets.models.backbone import BackBone, IntHyperparameter


class Transformer(BackBone):
    """Transformer backbone"""

    n_heads: IntHyperparameter
    n_layers: IntHyperparameter
    d_model: IntHyperparameter
    
    key_dim: IntHyperparameter
    feed_forward_hidden_dim: IntHyperparameter

    transformer: Transformer_nn

    def __setup__(self):
        self.transformer = Transformer_nn(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_model=self.d_model,
            key_dim=self.key_dim,
            feed_forward_hidden_dim=self.feed_forward_hidden_dim,
        )

    def forward(self, batch, return_reg_loss=False):
        del return_reg_loss # no regularization loss for transformer
        return self.transformer(batch), None

    def get_input_dim(self): return self.d_model
    def get_output_dim(self): return self.d_model


