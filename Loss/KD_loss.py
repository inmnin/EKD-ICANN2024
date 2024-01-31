import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class KDLoss(nn.Module):
    """PyTorch version of `ViTKD: Practical Guidelines for ViT feature knowledge distillation` """
    def __init__(self,BETA):
        self.T = BETA

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        preds_S and preds_T are both logit of output
        """

        #学生与老师的特征
        x_s = preds_S
        x_t = preds_T




        return loss_lr
