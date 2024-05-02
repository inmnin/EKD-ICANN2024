import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class EKDTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super(EKDTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_attention_heads)
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        attention_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
        out1 = self.norm1(hidden_states + attention_output)

        intermediate_output = self.intermediate(out1)
        layer_output = self.output(intermediate_output)
        out2 = self.norm2(out1 + layer_output)

        return out2


class EKDLoss(nn.Module):
    def __init__(self, student_hidden_size=312, teacher_hidden_size=768, num_layers=3, num_attention_heads=8,
                 intermediate_size=3072, max_seq_len=200, init_std=0.02):
        super(EKDLoss, self).__init__()

        self.align_student = nn.Linear(student_hidden_size, teacher_hidden_size)
        self.layers = nn.ModuleList(
            [EKDTransformerLayer(teacher_hidden_size, num_attention_heads, intermediate_size) for _ in
             range(num_layers)])
        self.loss_mse = nn.MSELoss()
        self.position_embeddings = nn.Embedding(512, teacher_hidden_size)  # Assuming a maximum length of 512
        self.sequence_length = max_seq_len
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, student_features_list, teacher_features_list, position_ids):
        B, N, D = student_features_list[0].size()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total_loss = 0
        lambda_threshold = 0.5  # Setting the Masking Probability
        position_ids = position_ids.int()
        position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.expand(B, self.sequence_length, 312).float()
        position_ids = self.align_student(position_ids)
        for student_features, teacher_features in zip(student_features_list, teacher_features_list):
            # Check and convert input to Tensor
            if not isinstance(student_features, torch.Tensor):
                student_features = torch.tensor(student_features)
            if not isinstance(teacher_features, torch.Tensor):
                teacher_features = torch.tensor(teacher_features)
            # Aligned feature dimensions

            aligned_student_features = self.align_student(student_features)
            aligned_teacher_features = teacher_features
            # Add positional embedding
            if position_ids is not None:
                # Get position embedding vector
                aligned_student_features += position_ids

            # Generate a random mask and record the masked position
            mask = torch.rand(aligned_student_features.size(1), device=device) < lambda_threshold
            masked_indices = mask.nonzero().squeeze()
            masked_student_features = aligned_student_features.clone()
            masked_student_features[:, mask, :] = 0 

            #  processing characteristics of students after alignment
            for layer in self.layers:
                masked_student_features = layer(masked_student_features.transpose(0, 1)).transpose(0, 1)

            student_features_masked = masked_student_features[:, masked_indices, :]
            teacher_features_masked = aligned_teacher_features[:, masked_indices, :]

            # Calculate the loss between student the representation constructed by the student and representation of the teacher 
            loss = self.loss_mse(student_features_masked, teacher_features_masked)
            total_loss += loss

        return total_loss



