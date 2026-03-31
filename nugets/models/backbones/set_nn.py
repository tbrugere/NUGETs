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

    d_model: int = int_hyperparameter(description="number of dimensions of the input and the output")

    embedding_layers: int=int_hyperparameter(description="number of layers for the embedding MLP")
    embedding_hidden_dim: int=int_hyperparameter(description="hidden dimension for embedding MLP")

    mlp: MLP = model_attribute()

    def __setup__(self):
        module_list = []
        module_list.append(nn.Linear(self.d_model, self.embedding_hidden_dim))
        module_list.append(nn.LeakyReLU())
        for _ in range(self.embedding_layers):
            module_list.append(nn.Linear(self.embedding_hidden_dim, self.embedding_hidden_dim))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(self.embedding_hidden_dim, self.d_model))
        self.mlp = nn.Sequential(*module_list)
        # self.mlp = MLP(self.d_model, 
        #                *[self.embedding_hidden_dim]*self.embedding_layers, 
        #                self.d_model, 
        #                batchnorm=False, 
        #                activation=nn.LeakyReLU)
        
    
    def forward(self, batch, return_reg_loss=False):
        del return_reg_loss 
        return self.mlp(batch), None

    def get_input_dim(self): return self.d_model
    def get_output_dim(self): return self.d_model
        

        


    