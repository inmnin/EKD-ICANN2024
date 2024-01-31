import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def Normalized_Mse_Loss(x_s, x_t):
    norm_x_s = torch.norm(x_s, p=2, dim=2, keepdim=True)
    norm_x_t = torch.norm(x_t, p=2, dim=2, keepdim=True)

    x_s_normalized = x_s / norm_x_s
    x_t_normalized = x_t / norm_x_t

    squared_diff = (x_s_normalized - x_t_normalized) ** 2

    mse = torch.sum(squared_diff)
    return mse


class PKDLoss(nn.Module):

    def __init__(self,
                 student_dims,
                 teacher_dims,
                 student_mim_layers,
                 teacher_mim_layers,
                 ):
        super(PKDLoss, self).__init__()
        self.student_kd_layers = student_mim_layers
        self.teacher_kd_layers = teacher_mim_layers
        self.num_kd_layers = len(student_mim_layers)
        if student_dims != teacher_dims:
            self.align = nn.ModuleList(
                [nn.Linear(student_dims, teacher_dims, bias=True) for i in range(self.num_kd_layers)])
        else:
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(List): [B*N*D,...,B*N*D], student's feature map
            preds_T(List): [B*N*D,...,B*N*D], teacher's feature map
        """

        x_s = preds_S
        x_t = preds_T

        B,_,_ = x_s[0].size()


        # print("align :", self.align[0].device)
        x_s = tuple(x_s[i] for i in self.student_kd_layers)
        x_t = tuple(x_t[i] for i in self.teacher_kd_layers)


        loss_mse = nn.MSELoss(reduction='sum')

        xc = []
        '''KD: Mimicking'''
        if self.align is not None:
            for i in range(self.num_kd_layers):

                xc.append(self.align[i](x_s[i]))

        else:
            xc = x_s

        loss_lr = 0
        for i in range(self.num_kd_layers):
            x_s_cls_state = xc[i][:, :1, :]
            x_t_cls_state = x_t[i][:, :1, :]
            loss_lr += Normalized_Mse_Loss(x_s_cls_state,x_t_cls_state)/B

        return loss_lr
