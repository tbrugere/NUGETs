from torch_heterogeneous_batching.nn.sumformer import GlobalEmbedding as Set_nn
from ml_lib.models.layers import MLP
from torch import nn

from nugets.models.backbone import BackBone, int_hyperparameter, hyperparameter, other_backbone_hyperparameter, model_attribute
from nugets.models.backbones.register import register

@register
class SetNN(BackBone):
    """
    Neural network for point sets: \phi(AGGR_{x \in P}(h(x)))
    Any aggregation can be used here: sum (DeepSets), min/max(PointNet)
    """

    input_dim: int = int_hyperparameter(description="number of dimensions of the input")
    output_dim: int = int_hyperparameter(description="dimension of the output")

    embedding_dim: int = int_hyperparameter(description="number of dimensions for the embedding MLP (latent dimension for h)")
    embedding_layers: int=int_hyperparameter(description="number of layers for the embedding MLP")
    embedding_hidden_dim: int=int_hyperparameter(description="hidden dimension for embedding MLP")
    aggregation: str = hyperparameter(default='mean', description="aggregation function")

    readout_hidden_dim: int=int_hyperparameter(description="hidden dimension for \phi")
    readout_layers: int=int_hyperparameter(description="number of layers for final readout MLP (\phi)")

    set_nn: Set_nn = model_attribute()
    readout_mlp: MLP = model_attribute()

    def __setup__(self):
        self.set_nn = Set_nn(input_dim=self.input_dim,
                             embeddimg_dim=self.embedding_dim,
                             hidden_dim=self.embedding_hidden_dim,
                             n_layers = self.embedding_layers,
                             aggregation=self.aggregation)
        self.readout_mlp = MLP(self.embedding_dim,
                               *[self.readout_hidden_dim]*self.readout_layers,
                               self.output_dim,
                               batchnorm=False, 
                               activation=nn.LeakyReLU,
                               end_activation=False)
    
    def forward(self, batch, return_reg_loss=False):
        del return_reg_loss 
        global_embedding = self.set_nn(batch)
        return self.readout_mlp(global_embedding)

    def get_input_dim(self): return self.input_dim
    def get_output_dim(self): return self.output_dim
        

        


    