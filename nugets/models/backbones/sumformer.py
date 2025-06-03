from torch_heterogeneous_batching.nn.sumformer import Sumformer as Sumformer_nn

from nugets.models.backbone import BackBone, int_hyperparameter, model_attribute
from nugets.models.backbones.register import register

@register
class Sumformer(BackBone):
    """
    SumFormer backbone
    """
    n_layers: int = int_hyperparameter(description="number of layers")
    d_model: int = int_hyperparameter(description="number of dimensions of the input and the output")

    feed_forward_hidden_dim: int = int_hyperparameter(description="hidden dimension")
    
    sumformer: Sumformer_nn = model_attribute()

    def __setup__(self):
        self.sumformer = Sumformer_nn(
            input_dim=self.d_model,
            hidden_dim = self.feed_forward_hidden_dim,
            num_blocks=self.n_layers
        )
    
    def forward(self, batch, return_reg_loss=False):
        del return_reg_loss # no regularization loss for SumFormers
        return self.sumformer(batch), None
    
    def get_input_dim(self): return self.d_model
    def get_output_dim(self): return self.d_model 