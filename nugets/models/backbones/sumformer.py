from torch_heterogeneous_batching.nn.sumformer import Sumformer as Sumformer_nn
from torch_geometric.nn.resolver import aggregation_resolver

from nugets.models.backbone import BackBone, int_hyperparameter,hyperparameter, model_attribute
from nugets.models.backbones.register import register

@register
class Sumformer(BackBone):
    """
    SumFormer backbone
    """
    n_layers: int = int_hyperparameter(description="number of layers")
    d_model: int = int_hyperparameter(description="number of dimensions of the input and the output")

    feed_forward_hidden_dim: int = int_hyperparameter(description="hidden dimension")

    aggregation: str = hyperparameter(type=str, description="sequence aggregation function", default='none')
    
    sumformer: Sumformer_nn = model_attribute()

    def __setup__(self):
        self.sumformer = Sumformer_nn(
            input_dim=self.d_model,
            hidden_dim = self.feed_forward_hidden_dim,
            num_blocks=self.n_layers
        )
        aggregation_args = {}
        if self.aggregation != "none":
            self.aggregation_fn = aggregation_resolver(self.aggregation, **aggregation_args)
    
    def forward(self, batch, return_reg_loss=False):
        del return_reg_loss # no regularization loss for SumFormers
        if self.aggregation != "none":
            return self.aggregation_fn(self.sumformer(batch).data, ptr=batch.ptr), None
        return self.sumformer(batch), None
    
    def get_input_dim(self): return self.d_model
    def get_output_dim(self): return self.d_model 