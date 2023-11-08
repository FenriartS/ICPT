import os.path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
from vqgan import get_vq_model

from mae_utils import generate_mask_for_evaluation

import math
from functools import reduce
from operator import mul
from torch.nn import Dropout
from torch.nn.modules.utils import _pair


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches # 196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # 1*1*1024
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False) # 1*197*1024  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.vae = get_vq_model().eval()
        vocab_size = 1024
        self.lam = 1
        
        # prompt_embeddings
        self.deep_prompt = True
        self.deep_prompt_dec = False
        self.num_tokens = 50
        self.prompt_dropout = Dropout(0.1)
        self.prompt_proj = nn.Identity()
        val = math.sqrt(6. / float(3 * reduce(mul, _pair(patch_size) , 1) + embed_dim))  # noqa
        val_dec = math.sqrt(6. / float(3 * reduce(mul, _pair(patch_size) , 1) + decoder_embed_dim))
        self.prompt_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        nn.init.uniform_(self.prompt_embed.data, -val, val)
        # self.prompt_embed_dec = nn.Parameter(torch.zeros(1, self.num_tokens, decoder_embed_dim))
        # nn.init.uniform_(self.prompt_embed_dec.data, -val_dec, val_dec)

        if self.deep_prompt:
            total_d_layer = depth - 1
            self.deep_prompt_embed = nn.Parameter(torch.zeros(
                total_d_layer, self.num_tokens, embed_dim))
            nn.init.uniform_(self.deep_prompt_embed.data, -val, val)

        if self.deep_prompt_dec:
            total_d_layer = decoder_depth - 1
            self.deep_prompt_embed_dec = nn.Parameter(torch.zeros(
                total_d_layer, self.num_tokens, decoder_embed_dim))
            nn.init.uniform_(self.deep_prompt_embed_dec.data, -val_dec, val_dec)
            # --------------------------------------------------------------------------

        self.prompt_embed_o = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.deep_prompt_embed_o = nn.Parameter(torch.zeros(total_d_layer, self.num_tokens, embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 512)
        )

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_prompt = None

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, vocab_size, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim   32, 196, 1024
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        if mask_ratio == 0:
            ids_shuffle, len_keep = generate_mask_for_evaluation()
            ids_shuffle = ids_shuffle.repeat(N, 1).to(x.device)
        ids_restore = torch.argsort(ids_shuffle, dim=1)   # 32*196

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))   # 32*49*1024

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)   # 32*196

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)   # bs*196*1024

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)   # bs*147*1024

        # img_p_idx, img_q_idx = [], []
        # for i in range(7):
        #     for j in range(7):
        #         img_p_idx.append(i * 14 + j)
        # img_q_idx = [i for i in range(7 * 14, 7 * 14 + 49)]
        # img_p_idx = torch.tensor(img_p_idx).repeat(x.shape[0], 1).to(x.device)
        # img_q_idx = torch.tensor(img_q_idx).repeat(x.shape[0], 1).to(x.device)
        # x_p = torch.gather(x, dim=1, index=img_p_idx.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        # x_q = torch.gather(x, dim=1, index=img_q_idx.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        # x_attn = self.img_attention(x_q, x_p, x_p)
        # x_pad = torch.zeros(x_attn.shape[0], self.num_tokens - x_attn.shape[1], x_attn.shape[2]).to(x_attn.device)
        # x_attn = torch.cat((x_attn, x_pad), dim=1).float()

        # x_extra = x[:, :98, :]
        # x_extra = torch.cat((x_extra[:, 1:, :], x_extra[:, :1, :]), dim=1)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)   # bs*1*1024
        x = torch.cat((cls_tokens, x), dim=1)      # bs*148*1024

        # append prompt tokens
        a = self.prompt_proj(self.prompt_embed).expand(x.shape[0] // 2, -1, -1)
        b = self.prompt_proj(self.prompt_embed_o).expand(x.shape[0] // 2, -1, -1)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(torch.cat((a, b), dim=0)),
                x[:, 1:, :]
            ), dim=1)   # bs*198*1024
        
        # apply Transformer blocks
        if self.deep_prompt:
            for i in range(len(self.blocks)):
                if i == 0:
                    x = self.blocks[i](x)
                else:
                    if i <= self.deep_prompt_embed.shape[0]:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embed[i-1]).expand(x.shape[0], -1, -1))
                        x = torch.cat((
                            x[:, :1, :],
                            deep_prompt_emb,
                            x[:, (1+self.num_tokens):, :]
                        ), dim=1)
                    x = self.blocks[i](x)
        else:
            for i, blk in enumerate(self.blocks):
                x = blk(x)
        x = self.norm(x)   # bs*148*1024

        self.mask_token_prompt = x[:, 1:50, :]
        x = torch.cat((x[:, :1, :], x[:, self.num_tokens + 1:, :]), dim=1)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)   # bs*148*512      

        # append mask tokens to sequence
        mask_tokens = self.mlp(self.mask_token_prompt) + self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # mask_tokens = self.mlp(self.mask_token_prompt).repeat(1, ids_restore.shape[1] + 1 - x.shape[1], 1) + self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)   # bs*49*512
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle  32*196*512
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # x = torch.cat((
        #         x[:, :1, :],
        #         self.prompt_dropout(self.prompt_proj(self.prompt_embed_dec).expand(x.shape[0], -1, -1)),
        #         x[:, 1:, :]
        #     ), dim=1) 

        # apply Transformer blocks
        if self.deep_prompt_dec:
            for i in range(len(self.decoder_blocks)):
                if i == 0:
                    x = self.decoder_blocks[i](x)
                else:
                    if i <= self.deep_prompt_embed_dec.shape[0]:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                            self.deep_prompt_embed_dec[i-1]).expand(x.shape[0], -1, -1))
                        x = torch.cat((
                                x[:, :1, :],
                                x[:, 1:(1+self.num_tokens), :],
                                x[:, (1+self.num_tokens):, :]
                            ), dim=1)   
                    x = self.decoder_blocks[i](x)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        loss_cont = self.forward_cont_loss(x[:, 1:, :])
        loss_kl = self.forward_kl_loss(x[:, 1:, :])

        # predictor projection
        x = self.decoder_pred(x)   # 32*197*1024

        # remove cls token
        x = x[:, 1:, :]   # 32*196*1024

        return x, loss_kl, loss_cont

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        with torch.no_grad():
            target = self.vae.get_codebook_indices(imgs).flatten(1)   # 16*196
        loss = nn.CrossEntropyLoss(reduction='none')(input=pred.permute(0, 2, 1), target=target)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward_kl_loss(self, x):
        z_i, z_j = x[:, :98, :], x[:, 98:, :]
        # representations = torch.cat([z_i, z_j], dim=0)
        logp_src_mem = F.log_softmax(z_j, dim=0)
        p_tgt_fus = F.softmax(z_i, dim=0)
        loss = F.kl_div(logp_src_mem, p_tgt_fus)

        return loss
    
    def forward_cont_loss(self, x):
        batch_size = x.shape[0]
        temperature = 0.5
        negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(x.device)).float()
        z_i, z_j = x[:, :98, :], x[:, 98:, :]
        z_i = torch.mean(z_i, dim=1)
        z_j = torch.mean(z_j, dim=1)
        representations = torch.cat([z_i[:4], z_j[:4], z_i[4:], z_j[4:]], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        positives = [similarity_matrix[i][j] for j in range(8) for i in range(8) if i != j]
        positives += [similarity_matrix[i][j] for j in range(8, 16) for i in range(8, 16) if i != j]
        positives = torch.stack(positives)
        
        nominator = torch.exp(positives / temperature)             # 2*bs
        denominator = negatives_mask * torch.exp(similarity_matrix / temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator))        # 2*bs
        loss = torch.sum(loss_partial) / 112

        return loss * 0.01

    def forward(self, imgs, visual_tokens=None, mask_ratio=0.75, inpt_mask=None):
        loss = {}
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred, loss['kl'], loss['cont'] = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        if visual_tokens is not None:
            loss['mae'] = self.forward_loss(visual_tokens, pred, mask)
        return loss, pred, mask


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        for i in range(4):
            nn.init.zeros_(self.linears[i].bias)
            nn.init.xavier_uniform_(self.linears[i].weight)
    
    def attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = self.attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.v_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.zeros_(self.v_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k):
        q = self.q_linear(q)
        k = self.k_linear(k)
        # k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], k.shape[1], self.num_heads, self.hidden_dim // self.num_heads).transpose(-2, -1)
        # kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bqcm->bqnm", qh * self.normalize_fact, kh)
        # x = weights

        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights


def mae_vit_small_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
