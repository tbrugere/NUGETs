from torch_heterogeneous_batching.nn.transformer import Transformer as Transformer_nn

from nugets.models.backbone import BackBone, int_hyperparameter, model_attribute


class Siamese(BackBone):
    """"
    Siamese network backbone

    Note that this network should take other backbone models as an argument

    If the Siamese network is used to learn a metric between two objects in the
    same space, model1=model2.
    However, I have added model1 and model2 to account for the case where we ask
    the Siamese network to learn something like Gromov-Wasserstein distance. In
    this case, this is an abuse of notation as the network is no longer "Siamese"
    but rather a "coupled" model. 
    
    """"
    encoder1: BackBone
    encoder2: BackBone

    decoder1: BackBone
    decoder2: BackBone
    p: int = int_hyperparameter(description="L_p distance function")
    decoder_distance: str = hyperparameter(type=str)
    
    def forward(self, batch, return_reg_loss=False):
        v1 = self.encoder1(batch.set1)
        v2 = self.encoder2(batch.set2)
        if return_reg_loss:
            out1 = decoder1(v1)
            out2 = decoder2(v2)
            reg = self.decoder_distance(out1, out2) #TODO: Change when we implement the loss module
        return torch.linalg.norm(v1 - v2, p=self.p)

    def get_input_dim(self): return (self.model1.get_input_dim(), self.model2.get_input_dim())
    def get_output_dim(self): return (self.model1.get_output_dim(), self.model2.get_output_dim())