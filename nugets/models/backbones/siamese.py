from torch_heterogeneous_batching.nn.transformer import Transformer as Transformer_nn

from nugets.models.backbone import BackBone, int_hyperparameter, model_attribute


class Siamese(BackBone):
    """"
    Siamese network backbone

    Note that this network should take other backbone models as an argument
    
    """"
    model: BackBone
    p: int = int_hyperparameter(description="")
    
    def forward(self, batch, return_reg_loss=False):
        v1 = self.model(batch.set1)
        v2 = self.model(batch.set2)
        return torch.linalg.norm(v1 - v2, p=self.p)

    def get_input_dim(self): return self.model.get_input_dim()
    def get_output_dim(self): return self.model.get_output_dim()

