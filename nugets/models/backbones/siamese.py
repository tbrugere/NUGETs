from typing import Callable
from ml_lib.models.layers import MLP
from torch_heterogeneous_batching import Batch
from torch_heterogeneous_batching.batch import BatchIndicator
from torch_heterogeneous_batching.nn.transformer import Transformer as Transformer_nn
import torch
from torch import nn

from nugets.models.backbone import (BackBone, int_hyperparameter, bool_hyperparameter, 
                model_attribute, hyperparameter,  other_backbone_hyperparameter, InnerBackbone)
from nugets.models.backbones.register import register
import nugets.losses.losses as Losses

from torch_geometric.nn.resolver import aggregation_resolver


@register
class CoupledNetwork(BackBone):
    """
    Siamese network backbone

    Note that this network should take other backbone models as an argument

    If the Siamese network is used to learn a metric between two objects in the
    same space, model1=model2.
    However, I have added model1 and model2 to account for the case where we ask
    the Siamese network to learn something like Gromov-Wasserstein distance. In
    this case, this is an abuse of notation as the network is no longer Siamese
    but rather a coupled model. 
    
    """

    encoder: InnerBackbone = other_backbone_hyperparameter("backbone for the encoder")

    reconstruct_input: bool = bool_hyperparameter(default=True, description="use reconstruction loss")

    decoder_hidden_dim: int = int_hyperparameter(default=512, description="backbone for the decoder" )
    decoder_n_layers: int = int_hyperparameter(default=3, description="number of layers of the decoder MLP")
    decoder_n_points: int = int_hyperparameter(default=50, description="number of points output by the decoder")

    latent_dimension: int = int_hyperparameter(description="dimension of the latent space, set it to 0 to disable latent")
    p: int = int_hyperparameter(description="L_p distance function")

    aggregation: str = hyperparameter(default='mean', type=str, description="aggregation function")

    decoder_distance: str = hyperparameter(type=str)

    encoder1: BackBone = model_attribute()
    encoder2: BackBone = model_attribute()

    encoder_projection_1: nn.Module = model_attribute()
    encoder_projection_2: nn.Module = model_attribute()

    decoder1: MLP|None = model_attribute()
    decoder2: MLP|None = model_attribute()

    decoder_loss_fn: Callable = model_attribute()

    def __setup__(self):
        self.encoder1 = self.encoder.load()
        self.encoder2 = self.encoder.load()
        
        if self.reconstruct_input:
            self.decoder1 = MLP(self.latent_dimension, 
                                *[self.decoder_hidden_dim]*self.decoder_n_layers, 
                                self.decoder_n_points * self.encoder1.get_input_dim())
            self.decoder2 = MLP(self.latent_dimension, 
                                *[self.decoder_hidden_dim]*self.decoder_n_layers, 
                                self.decoder_n_points * self.encoder1.get_input_dim())
        else: self.decoder1 = self.decoder2 = None

        if self.latent_dimension:
            self.encoder_projection_1 = nn.Linear(self.encoder1.get_output_dim(), self.latent_dimension)
            self.encoder_projection_2 = nn.Linear(self.encoder2.get_output_dim(), self.latent_dimension)
        else: 
            self.encoder_projection_1 = nn.Identity()
            self.encoder_projection_2 = nn.Identity()

        self.decoder_loss_fn = getattr(Losses, self.decoder_distance)()

        if self.aggregation == "none":
            raise ValueError('Aggregation function must not be none for Coupled or Siamese networks')
        aggregation_args = {}
        self.aggregation_fn = aggregation_resolver(self.aggregation, **aggregation_args)

            
    def forward(self, batch, return_reg_loss=False):
        set1, set2 = batch
        batch_size = set1.batch_size
        v1, _ = self.encoder1(set1) # for now, ignore regularization for encoders/decoders
        v2, _ = self.encoder2(set2)

        # v1 = self.aggregation_fn(v1.data, ptr=v1.ptr)
        # v2 = self.aggregation_fn(v2.data, ptr=v2.ptr)
        v1 = self.encoder_projection_1(v1.mean())
        v2 = self.encoder_projection_1(v2.mean())

        v1 = self.encoder_projection_1(v1)
        v2 = self.encoder_projection_2(v2)

        predicted_distances = torch.linalg.vector_norm(v1 - v2, ord=self.p, dim=-1)


        if self.reconstruct_input and self.training:
            assert self.decoder1 is not None and self.decoder2 is not None
            out1_data = self.decoder1(v1)\
                    .reshape(batch_size * self.decoder_n_points, self.encoder1.get_input_dim())
            out2_data = self.decoder2(v2)\
                    .reshape(batch_size * self.decoder_n_points, self.encoder2.get_input_dim())
            indicator = BatchIndicator(torch.full((batch_size,), self.decoder_n_points, device=v1.device))
            out1 = Batch(out1_data, order=1, indicator=indicator)
            out2 = Batch(out2_data, order=1, indicator=indicator)
            return (predicted_distances, out1, out2), None
        elif self.reconstruct_input: 
            return (predicted_distances, None, None), None
        else: 
            return predicted_distances, None

    def get_input_dim(self): 
        return (self.encoder1.get_input_dim(), self.encoder2.get_input_dim())
    def get_output_dim(self): 
        out_dim = self.encoder1.get_output_dim()
        assert out_dim == self.encoder2.get_output_dim() 
        return out_dim

@register
class Siamese(CoupledNetwork):
    encoder: InnerBackbone = other_backbone_hyperparameter("backbone for the encoder")
    decoder: InnerBackbone = other_backbone_hyperparameter("backbone for the decoder")

    aggregation: str = hyperparameter(default='mean', type=str, description="aggregation function")
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
        self.encoder1 = self.encoder2 = self.encoder.load()
        self.decoder1 = self.decoder2 = self.decoder.load()

        if self.latent_dimension:
            self.encoder_projection_1 = self.encoder_projection_2 = nn.Linear(self.encoder1.get_output_dim(), self.latent_dimension)
            self.decoder_projection_1 = self.decoder_projection_2  = nn.Linear(self.latent_dimension, self.decoder2.get_input_dim())
        else: 
            self.encoder_projection_1 = self.encoder_projection_2 = nn.Identity()
            self.decoder_projection_1 = self.decoder_projection_2 = nn.Identity()
