from Loss.regressor import *
from functools import partial

#创建位置编码


class ContextAutoencoderLoss(nn.Module):
    def __init__(self, student_dims=768, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None, init_std=0.02,
                 teacher_dims=768, regressor_depth=4, regressor_num_heads=12,
                 regressor_layer_scale_init_value=0.1, fix_init_weight=False, mask_rate = 0.3):
        super().__init__()

        self.student_dims = student_dims
        self.teacher_dims = teacher_dims
        self.init_std = init_std
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        #学生特征中各token被遮蔽的概率
        self.mask_rate = mask_rate


        # from encoder to regresser projection, borrowed from mae.
        if teacher_dims != student_dims:
            self.encoder_to_regresser = nn.Linear(student_dims, teacher_dims, bias=True)
            self.encoder_to_regresser_norm = norm_layer(teacher_dims)
        else:
            self.encoder_to_regresser = None

        # context regresser
        self.regresser = LatentRegresser(embed_dim=teacher_dims, regresser_depth=regressor_depth,
                                         num_heads=regressor_num_heads,
                                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                         attn_drop_rate=attn_drop_rate,
                                         drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                                         init_values=regressor_layer_scale_init_value, init_std=init_std,
                                         )

        # regress is cross attention, mask tokens are querries.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))
        trunc_normal_(self.mask_token, std=self.init_std)


            ### whether to use 'rescale' to init the weight, borrowed from beit.
        if not fix_init_weight:
            self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    '''
    Input shape:
    '''

    def forward(self, xs, xt, pos_embed):


        unmasked_cls_state_s = xs[:, :1, :]
        unmasked_hidden_state_s = xs[:, 1:, :]

        B,N,D = unmasked_hidden_state_s.size()

        # 创建一个形状为 [B, N] 的布尔张量，所有元素初始化为 False
        bool_masked_pos = torch.zeros(B, N, dtype=torch.bool, device=xs.device)

        # 对每个batch中的每个句子随机选择K个不重复的位置，并将这些位置设为True
        K = int(self.mask_rate * N)
        for batch_idx in range(B):
            random_indices = torch.randperm(N)[:K]
            bool_masked_pos[batch_idx, random_indices] = True


        # 确保 bool_masked_pos 也在 CUDA 上
        bool_masked_pos = bool_masked_pos.to(unmasked_hidden_state_s.device)

        xs_unmasked = unmasked_hidden_state_s[~bool_masked_pos].reshape(B,-1,D)
        xs_unmasked = torch.cat((unmasked_cls_state_s, xs_unmasked), dim=1)

        '''
        Encoder
        Output shape:
            [bs, num_visible + 1, C]
        '''

        # encoder to regresser projection
        xs_unmasked = self.encoder_to_regresser(xs_unmasked)
        xs_unmasked = self.encoder_to_regresser_norm(xs_unmasked)
        pos_embed = self.encoder_to_regresser(pos_embed)
        pos_embed = self.encoder_to_regresser_norm(pos_embed)

        '''
        Alignment branch
        '''

        with torch.no_grad():
            cls_state_t = xt[:, :1, :]
            xt_masked = xt[:, 1:, :][bool_masked_pos].reshape(B, -1, self.teacher_dims)
            latent_target = torch.cat((cls_state_t,xt_masked),dim=1)

        '''
        Latent contextual regressor
        1. prepare masked, unmasked pos embed, and masked mebedding
        '''


        unmasked_cls_state_s = xs_unmasked[:,:1,:]
        unmasked_hidden_state_s = xs_unmasked[:,1:,:]

        _, num_unmasked, dim = unmasked_hidden_state_s.shape

        pos_embed = pos_embed.unsqueeze(0)
        pos_embed = pos_embed[:,:N+1,:]

        tranc_pos_embed = pos_embed.expand(B, N+1, self.teacher_dims).cuda(xs_unmasked.device)
        pos_embed_masked = tranc_pos_embed[:, 1:,:][bool_masked_pos].reshape(B, -1,
                                                                     self.teacher_dims)  # pos embed for masked patches
        pos_embed_unmasked = tranc_pos_embed[:, 1:,:][~bool_masked_pos].reshape(B, -1,
                                                                     self.teacher_dims)  # pos embed for unmasked patches

        num_masked_tokens = N - num_unmasked


        xs_masked = self.mask_token.expand(B, num_masked_tokens, -1)  # masked embedding

        '''
        2. regress masked latent via regresser
        '''

        xs_masked_predict = self.regresser(xs_masked, xs_unmasked[:,1:,:], pos_embed_masked, pos_embed_unmasked)

        ## preserve for alignment
        latent_predict = torch.cat((cls_state_t,xs_masked_predict),dim=1)

        loss_mse = nn.MSELoss(reduction='sum')
        loss_lr = loss_mse(latent_predict, latent_target) / B

        return loss_lr