from typing import Set
from torch.utils.data import DataLoader

import lightning as pl

from nugets.models import BackBone, Model
from nugets.models.backbones.transformer import Transformer
from nugets.tasks.dummy_tasks import SetIdentityTask
