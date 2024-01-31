import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TINYLoss(nn.Module):

    def __init__(self,
                 student_dims,
                 teacher_dims,
                 student_mim_layers,
                 teacher_mim_layers
                 ):
        super(TINYLoss, self).__init__()

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
                preds_T,
                student_attentions,
                teacher_attentions
    ):
        
        x_s = preds_S
        x_t = preds_T

        B,_,_ = x_s[0].size()


        # print("align :", self.align[0].device)
        x_s = tuple(x_s[i] for i in self.student_kd_layers)
        x_t = tuple(x_t[i] for i in self.teacher_kd_layers)
        x_s_attention = tuple(student_attentions[i-1] for i in self.student_kd_layers[1:])
        x_t_attention = tuple(teacher_attentions[i-1] for i in self.teacher_kd_layers[1:])

        x_c = []
        '''KD: Mimicking'''
        if self.align is not None:
            for i in range(self.num_kd_layers):

                x_c.append(self.align[i](x_s[i]))

        else:
            x_c = x_s


        loss_mse = nn.MSELoss()

        loss_emb = loss_mse(x_c[0],x_t[0])

        loss_hid = 0
        for i in range(1,self.num_kd_layers):
            loss_hid += loss_mse(x_c[i],x_t[i])
        loss_hid /= self.num_kd_layers-1

        loss_atten = 0
        for student_attention, teacher_attention in zip(x_s_attention, x_t_attention):
            student_attention_mean = student_attention.mean(dim=1)
            teacher_attention_mean = teacher_attention.mean(dim=1)
            loss_atten += loss_mse(student_attention_mean, teacher_attention_mean)

        loss_atten /= self.num_kd_layers - 1

        loss = loss_emb + loss_hid + loss_atten
        return loss

