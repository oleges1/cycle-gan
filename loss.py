import torch
from torch import nn

criterionCycle = nn.L1Loss()
D_loss = nn.BCEWithLogitsLoss()
