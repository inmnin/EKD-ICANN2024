import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MINLMLoss(nn.Module):

    def __init__(self,
                 student_dims,
                 teacher_dims,
                 student_mim_layers,
                 teacher_mim_layers,
                 student,
                 teacher,
                 L: int,
                 M: int,
                 A_r: int,
                 alpha,
                 beta,
                 gama,
                 train_type
                 ):
        super(MINLMLoss, self).__init__()

        self.student_kd_layers = student_mim_layers
        self.teacher_kd_layers = teacher_mim_layers
        self.num_kd_layers = len(student_mim_layers)
        if student_dims != teacher_dims:
            self.align = nn.ModuleList(
                [nn.Linear(student_dims, teacher_dims, bias=True) for i in range(self.num_kd_layers)])
        else:
            self.align = None

        self.teacher = teacher
        self.student = student
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="sum")
        self.teacher.eval()
        self.student.train()
        self.L = L
        self.M = M
        self.A_r = A_r
        
        if train_type == "minlmv1":
            self.relations = {(1,2):1/2 , (3,3):1/2}
            
        if train_type == "minlmv2":
            self.relations = {(1,1):1/3 ,(2,2):1/3, (3,3):1/3}
        # Make sure not updating teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.alpha = alpha
        self.beta = beta
        self.gama = gama

    def _get_relation_vectors(self, self_attn, prev_hidden, relation_head_size: int):
        """Get query, key, and value of relation heads of the last attention layer.

        The vectors' shape will be (batch_size, relation_head_number, seq_length, relation_head_size).
        """
        q = self._transpose_for_scores_relation(
            self_attn.query(prev_hidden), relation_head_size
        )
        k = self._transpose_for_scores_relation(
            self_attn.key(prev_hidden), relation_head_size
        )
        v = self._transpose_for_scores_relation(
            self_attn.value(prev_hidden), relation_head_size
        )
        return q, k, v

    def _transpose_for_scores_relation(self, x: torch.Tensor, relation_head_size: int):
        """Adapted from BertSelfAttention.get_transposed_attns().

        Arguments:
            x (Tensor): a vector (query, key, or value) of shape (batch_size, seq_length, hidden_size)
            relation_head_size (int): relation head size
        Return:
            x_relation (Tensor): a vector (query, key, or value) of shape
                                (batch_size, relation_head_number, seq_length, relation_head_size)
        """
        new_x_shape = [*x.size()[:-1], self.A_r, relation_head_size]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _get_kl_loss(
            self, rel_T: torch.Tensor, rel_S: torch.Tensor, attention_mask: torch.Tensor
    ):
        """Compute KL divergence loss of teacher and student on one relation.

        Arguments:
            rel_T: a self attention relation of the teacher (batch_size, A_r, seq_len, seq_len)
            rel_S: a self attention relation of the student (batch_size, A_r, seq_len, seq_len)
            attention_mask: attention mask of a batch of input
        """
        # Note: rel_T is the target and rel_S is the input of KL Div loss for KLDivLoss(), before softmax.
        # KLDivLoss() needs log of inputs (rel_S)
        loss = 0.0
        batch_size = attention_mask.shape[0]
        seq_lengths = attention_mask.sum(-1).tolist()
        for b in range(batch_size):
            cur_seq_len = seq_lengths[b]  # current sequence length
            R_L_T = torch.nn.Softmax(dim=-1)(rel_T[b, :, :cur_seq_len, :cur_seq_len])
            R_M_S = torch.nn.functional.log_softmax(
                rel_S[b, :, :cur_seq_len, :cur_seq_len], dim=-1
            )  # KL DIV loss needs log, so do log_softmax
            loss += self.kl_loss_fn(
                R_M_S.reshape(-1, cur_seq_len), R_L_T.reshape(-1, cur_seq_len)
            ) / (
                            self.A_r * cur_seq_len
                    )  # normalize by relation head num and seq length
        loss /= batch_size  # normalize by batch_size as well
        return loss

    def forward(self, input_ids, attention_mask,labels):
        """Run a forward pass over the input. Return a tuple of one element, which is the MiniLM loss.

        Note: the return value is a tuple since HuggingFace trainer uses outputs[0] as loss.

        Arguments:
            input_ids: input_id tokens for a batch.
            token_type_ids: token_type_ids (indicating sentence_id) for a batch.
            attention_mask: Attention mask (indicating which tokens are pad tokens) for a batcb.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        student_outs = self.student(
            **inputs, output_hidden_states=True, output_attentions=True
        )
        
        teacher_outs = self.teacher(
            **inputs, output_hidden_states=True, output_attentions=True
        )


        L = self.L  # layer to distill from in teacher (can be any teacher layer)
        M = self.M  # layer to distill to in student (last layer)

        d_h_T = self.teacher.config.hidden_size  # teacher's hidden size
        d_h_S = self.student.config.hidden_size  # student's hidden size
        d_r_T = d_h_T // self.A_r  # teacher's relation head size
        d_r_S = d_h_S // self.A_r  # student's relation head size

        # hidden_states contains L+1 elements for the teacher and M+1 elements for the student,
        # since the first is embedding
        # To calculate query, key, and value for the last attention layer, we get the hidden states
        # of the second last layer (L+1 -2 = L - 1)
        hidden_L_1_T = teacher_outs.hidden_states[L - 1]
        hidden_M_1_S = student_outs.hidden_states[M - 1]

        # Get relation vectors (query, key, value) of the shape (batch_size, A_r, seq_len, d_r) based on Figure 1
        relation_vectors_T = self._get_relation_vectors(
            self.teacher.bert.encoder.layer[L - 1].attention.self, hidden_L_1_T, d_r_T
        )
        relation_vectors_S = self._get_relation_vectors(
            self.student.bert.encoder.layer[M - 1].attention.self, hidden_M_1_S, d_r_S
        )

        loss = 0  # total loss of all types of relations
        for relation_pair, weight in self.relations.items():
            # Calculate loss for each pairs of relations
            # 1-> Query, 2-> Key, 3-> Value.
            # relation pair of (1,2) indicates to compute QK for teacher and student and apply loss on it
            m, n = relation_pair  # m and n are 1-indexed

            # Formula (7) and (8)
            A_L_T_scaleddot = torch.matmul(
                relation_vectors_T[m - 1], relation_vectors_T[n - 1].transpose(-1, -2)
            ) / math.sqrt(
                d_r_T
            )  # (batch_size, A_r, seq_len, seq_len)
            A_M_S_scaleddot = torch.matmul(
                relation_vectors_S[m - 1], relation_vectors_S[n - 1].transpose(-1, -2)
            ) / math.sqrt(d_r_S)

            # Compute relaiton loss (Formula (6))
            l_relation = self._get_kl_loss(
                A_L_T_scaleddot.detach(), A_M_S_scaleddot, inputs["attention_mask"]
            )

            
            # Aggregate losses (Formula (5))
            loss += weight * l_relation

        
        s_loss = student_outs.loss
        
        c_loss = 0
        if self.beta>0:
            
            s_logits = student_outs.logits
            t_logits = teacher_outs.logits

            KL = nn.KLDivLoss(reduction='batchmean')  
            t_logits_soft = F.softmax(t_logits / self.beta, dim=1)
            s_logits_soft = F.log_softmax(s_logits / self.beta, dim=1)
            c_loss = KL(s_logits_soft, t_logits_soft)*(self.beta*self.beta)
        
        
        f_loss = (self.gama)*loss+(self.alpha)*c_loss+(1-self.alpha)*s_loss
        return f_loss

