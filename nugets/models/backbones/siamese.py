from torch_heterogeneous_batching.nn.transformer import Transformer as Transformer_nn
import torch
from torch import nn

from nugets.models.backbone import (BackBone, int_hyperparameter, 
                model_attribute, hyperparameter,  other_backbone_hyperparameter, InnerBackbone)
from nugets.models.backbones.register import register


@register
class CoupledNetwork(BackBone):
    """
    Siamese network backbone

    Note that this network should take other backbone models as an argument

    If the Siamese network is used to learn a metric between two objects in the
    same space, model1=model2.
    However, I have added model1 and model2 to account for the case where we ask
    the Siamese network to learn something like Gromov-Wasserstein distance. In
    this case, this is an abuse of notation as the network is no longer "Siamese"
    but rather a "coupled" model. 
    
    """

    encoder: InnerBackbone = other_backbone_hyperparameter("backbone for the encoder")
    decoder: InnerBackbone = other_backbone_hyperparameter("backbone for the decoder")


    latent_dimension: int = int_hyperparameter(description="dimension of the latent space, set it to 0 to disable latent")
    p: int = int_hyperparameter(description="L_p distance function")
    decoder_distance: str = hyperparameter(type=str)

    encoder1: BackBone = model_attribute()
    encoder2: BackBone = model_attribute()

    encoder_projection_1: nn.Module = model_attribute()
    encoder_projection_2: nn.Module = model_attribute()
    decoder_projection_1: nn.Module = model_attribute()
    decoder_projection_2: nn.Module = model_attribute()

    decoder1: BackBone = model_attribute()
    decoder2: BackBone = model_attribute()


    def __setup__(self):
        self.encoder1 = self.encoder.load()
        self.encoder2 = self.encoder.load()
        self.decoder1 = self.decoder.load()
        self.decoder2 = self.decoder.load()

        if self.latent_dimension:
            self.encoder_projection_1 = nn.Linear(self.encoder1.get_output_dim(), self.latent_dimension)
            self.encoder_projection_2 = nn.Linear(self.encoder2.get_output_dim(), self.latent_dimension)
            self.decoder_projection_1 = nn.Linear(self.latent_dimension, self.decoder2.get_input_dim())
            self.decoder_projection_2 = nn.Linear(self.latent_dimension, self.decoder2.get_input_dim())
        else: 
            self.encoder_projection_1 = nn.Identity()
            self.encoder_projection_2 = nn.Identity()
            self.decoder_projection_1 = nn.Identity()
            self.decoder_projection_2 = nn.Identity()

            

    
    def forward(self, batch, return_reg_loss=False):
        v1 = self.encoder_projection_1(self.encoder1(batch.set1))
        v2 = self.encoder_projection_2(self.encoder2(batch.set2))
        if return_reg_loss:
            out1 = self.decoder1(self.decoder_projection_1(v1))
            out2 = self.decoder2(self.decoder_projection_2(v2))
            reg = self.decoder_distance(batch.set1, out1) #TODO: Change when we implement the loss module
            reg = reg + self.decoder_distance(batch.set2, out2)
        else: reg = None
        return torch.linalg.norm(v1 - v2, p=self.p), reg

    def get_input_dim(self): 
        return (self.encoder1.get_input_dim(), self.encoder2.get_input_dim())
    def get_output_dim(self): 
        return (self.encoder1.get_output_dim(), self.encoder2.get_output_dim())

class Siamese(CoupledNetwork):
    pass
