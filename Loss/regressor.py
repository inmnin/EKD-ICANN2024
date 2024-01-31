import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Loss.modeling_finetune import DropPath, Mlp
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


'''
Modified from Attention()
'''

class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


#Regressor由以下block块构成
class RegressorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp_cross = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1_cross = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2_cross = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1_cross = nn.Parameter(torch.ones((dim)), requires_grad=False)
            self.gamma_2_cross = nn.Parameter(torch.ones((dim)), requires_grad=False)

    def forward(self, x_q, x_kv, pos_q, pos_k):
        x = x_q + self.drop_path(
            self.gamma_1_cross * self.cross_attn(self.norm1_q(x_q + pos_q), k=self.norm1_k(x_kv + pos_k),
                                                 v=self.norm1_v(x_kv)))
        x = self.norm2_cross(x)
        x = x + self.drop_path(self.gamma_2_cross * self.mlp_cross(x))

        return x


'''
Latent context regressor that regresses masked representation 
from unmasked representation using cross attention
'''


class LatentRegresser(nn.Module):
    def __init__(self, embed_dim=768, regresser_depth=6, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, init_std=0.02, model_type='cae'):
        super().__init__()
        self.model_type = model_type

        self.num_features = self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, regresser_depth)]

        # context regressor
        self.regressor_blocks = nn.ModuleList([
            RegressorBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(regresser_depth)])

        self.norm = norm_layer(embed_dim)

        self.init_std = init_std

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.regressor_blocks):
            rescale(layer.cross_attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp_cross.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x_masked, x_unmasked, pos_embed_masked, pos_embed_unmasked):

        # latent contextual regressor: regress masked latent from unmask latent
        for blk in self.regressor_blocks:
            x_masked = blk(x_masked, torch.cat([x_unmasked, x_masked], dim=1), pos_embed_masked,
                           torch.cat([pos_embed_unmasked, pos_embed_masked], dim=1))


        x_masked = self.norm(x_masked)

        return x_masked