import mamba_ssm
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class spa_patchfy(nn.Module):
    def __init__(self, spa_size, patch_size, channel, D_model):
        super(spa_patchfy, self).__init__()
        self.channel = channel
        self.spa_size = spa_size
        self.patch_size = patch_size
        self.Dyipie = self.channel * self.spa_size * self.spa_size
        if self.patch_size == 15:
            self.mini_patch_size = self.patch_size // self.spa_size
            self.proj = nn.Conv2d(self.channel, self.Dyipie, self.spa_size, self.spa_size)
        else:
            self.mini_patch_size = self.patch_size // self.spa_size + 1
            self.proj = nn.Conv2d(self.channel, self.Dyipie, self.spa_size, self.spa_size, padding=1)

        self.D_model = D_model
        self.Espa = nn.Conv1d(self.Dyipie, self.D_model, 1, 1)
        self.positional_embedding = nn.init.normal_(self.get_2d_positional_encoding(self.mini_patch_size,
                                                                                    self.Dyipie), std=0.01)

    def get_2d_positional_encoding(self, L, D):
        assert D % 2 == 0, "Dimension must be even"
        positional_encoding = torch.zeros(L, L, D).cuda(0)

        for i in range(L):
            for j in range(L):
                for d in range(D):
                    if d % 2 == 0:
                        # 对于偶数索引d，使用sin函数
                        positional_encoding[i, j, d] = math.sin(i / (10000 ** (d / D))) + math.sin(
                            j / (10000 ** ((d + 1) / D)))
                    else:
                        # 对于奇数索引d，使用cos函数
                        positional_encoding[i, j, d] = math.cos(i / (10000 ** ((d - 1) / D))) + math.cos(
                            j / (10000 ** (d / D)))

        return positional_encoding

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.positional_embedding.cuda(0) + x
        x = x.permute(0, 3, 1, 2)
        # add 2d positional embedding
        x = x.flatten(2)  # BCHW -> BCM
        x = self.Espa(x)
        x = x.transpose(1, 2)
        return x

class spe_patchfy(nn.Module):
    def __init__(self, spe_size, patch_size, channel, D_model, center_size=3):
        super(spe_patchfy, self).__init__()
        self.patch_size = patch_size
        self.spe_size = spe_size
        self.center_size = center_size
        self.spamap_size = center_size*center_size
        self.Dyipie = self.spamap_size*self.spe_size
        self.D_model = D_model
        self.channel = channel
        self.proj = nn.Conv1d(self.spamap_size, self.Dyipie, self.spe_size, self.spe_size)
        self.Espe = nn.Conv1d(self.Dyipie, self.D_model,1,1)
        self.positional_embedding = nn.init.normal_(nn.Parameter(torch.empty(self.channel//self.spe_size, self.D_model)), std=0.01)

    def from_front_to_back(self, x):
        return x

    def from_back_to_front(self, x):
        x = x.flip(dims=[1])
        return x

    def forward(self, x):
        start = self.patch_size // 2 - 1
        end = start + 3
        x = x[:, :, start:end, start:end]
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.proj(x) # BCHW -> BCN
        x = self.Espe(x)
        x = x.transpose(1, 2)
        x = x + self.positional_embedding
        # add 1d positional embedding
        # x = (self.from_front_to_back(x) + self.from_back_to_front(x))/2
        return x

class spa_spe_star_fusion(nn.Module):
    def __init__(self, ind_model, outd_model):
        super(spa_spe_star_fusion, self).__init__()
        self.ind_model = ind_model
        self.outd_model = outd_model
        self.starlinear1 = nn.Linear(self.ind_model, self.ind_model)
        self.starlinear2 = nn.Linear(self.ind_model, self.ind_model)

        self.conv1 = nn.Conv1d(self.ind_model, self.outd_model, 1, 1)
        self.conv2 = nn.Conv1d(self.ind_model, self.outd_model, 1, 1)
        self.ln1 = nn.LayerNorm(self.outd_model)
        self.ln2 = nn.LayerNorm(self.outd_model)
        self.act = nn.ReLU()


    def forward(self, spa_token, spe_token):
        N = spa_token.size(1)
        M = spe_token.size(1)
        spa_center_token = spa_token[:, spa_token.size(1)//2 + 1, :]
        spe_mean_token = spe_token.mean(dim=1)
        ss_token = spa_center_token + spe_mean_token
        ss_token1 = self.starlinear1(ss_token)
        ss_token2 = self.starlinear2(ss_token)
        ss_token = ss_token1 * ss_token2
        ss_token = self.act(ss_token)
        ss_token = ss_token.unsqueeze(1)
        spa_token = spa_token * ss_token.repeat(1, N, 1)
        spa_token = self.conv1(spa_token.transpose(1, 2)).transpose(1, 2)
        spe_token = spe_token * ss_token.repeat(1, M, 1)
        spe_token = self.conv2(spe_token.transpose(1, 2)).transpose(1, 2)
        spa_token = self.ln1(spa_token)
        spe_token = self.ln2(spe_token)
        return spa_token, spe_token


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: str = 'cuda'):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class Mamba_Block(nn.Module):
    def __init__(self, d_model=64, seq_length=77):
        super(Mamba_Block, self).__init__()
        self.d_model = d_model
        self.sqe_length = seq_length

        self.norm1 = RMSNorm(d_model=self.d_model)
        self.adapool = nn.AdaptiveAvgPool1d(self.sqe_length)
        self.mamba = mamba_ssm.Mamba(d_model=self.d_model, device=0)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.linear1 = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        x = self.adapool(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm1(x) + self.linear1(self.mamba(self.norm1(x)))
        return x

class Mamba_Mask_Block(nn.Module):
    def __init__(self, d_model=64, D_yipie=0):
        super(Mamba_Mask_Block, self).__init__()
        self.d_model = d_model
        self.D_yipie = D_yipie
        self.adapool = nn.AdaptiveAvgPool1d(self.D_yipie)

        self.norm1 = RMSNorm(d_model=self.d_model)
        self.mamba = mamba_ssm.Mamba_Mask(d_model=self.d_model, device=0)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.linear1 = nn.Linear(self.d_model, self.d_model)
        self.linear2 = nn.Linear(self.d_model, self.d_model)
        self.linear3 = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        x = self.adapool(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.norm1(x) + self.linear1(self.mamba(self.norm1(x)))
        return x


class big_mamba_block(nn.Module):
    def __init__(self, ind_model, outd_model, spa_seqlen, spe_seqlen):
        super(big_mamba_block, self).__init__()
        self.ind_model = ind_model
        self.outd_model = outd_model
        self.spa_seqlen = spa_seqlen
        self.spe_seqlen = spe_seqlen
        self.spa_mamba = Mamba_Mask_Block(ind_model, spa_seqlen)
        self.spe_mamba = Mamba_Mask_Block(ind_model, spe_seqlen)
        self.ss_star_fusion = spa_spe_star_fusion(self.ind_model, self.outd_model)
        self.linear1 = nn.Linear(self.ind_model, self.outd_model)
        self.linear2 = nn.Linear(self.ind_model, self.outd_model)

    def forward(self, spa_token, spe_token):
        x = spa_token
        y = spe_token
        spa_token = self.spa_mamba(spa_token)
        spe_token = self.spe_mamba(spe_token)
        spa_token, spe_token = self.ss_star_fusion(spa_token, spe_token)
        spa_token = spa_token + F.silu(self.linear1(x))
        spe_token = spe_token + F.silu(self.linear2(y))
        return spa_token, spe_token


class ss_linear_block(nn.Module):
    def __init__(self, d_model,end=0, emb_dropout=0.):
        super(ss_linear_block, self).__init__()
        self.d_model = d_model
        self.end = end
        self.bnspa = nn.BatchNorm1d(self.d_model)
        self.bnspe = nn.BatchNorm1d(self.d_model)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Linear(self.d_model, 7)
        )
        self.pro_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(emb_dropout),
            nn.Linear(self.d_model, 256)
        )

    def forward(self, spa_token, spe_token):
        if self.end == 0:
            spa_token = spa_token[:, spa_token.size(1) // 2 + 1, :]
            spa_token = self.bnspa(spa_token)
            spe_token = spe_token.mean(dim=1)
            spe_token = self.bnspe(spe_token)
            result = spa_token + spe_token
            result = self.to_latent(result)
            return self.mlp_head(result)
        else:
            spa_token = spa_token[:, spa_token.size(1) // 2 + 1, :]
            spa_token = self.bnspa(spa_token)
            spe_token = spe_token.mean(dim=1)
            spe_token = self.bnspe(spe_token)
            result = spa_token + spe_token
            result = self.to_latent(result)
            return self.mlp_head(result), self.pro_head(result)

class CSMamba(nn.Module):
    def __init__(self, context_length=77, vocab_size=49408, batch_size=256, n_bands=48, patch_size=27, emb_dropout=0, spa_size=3, spe_size=4, layer_d_model=[64, 64, 32, 16, 8]):
        super(CSMamba, self).__init__()
        self.d_model = 64
        self.center_size = 3
        self.spa_size = spa_size
        self.spe_size = spe_size
        self.patch_size = patch_size
        self.inchannel = 220
        self.n_bands = 110
        self.layer_d_model = layer_d_model
        if self.patch_size == 15:
            self.spa_seqlen = (self.patch_size//self.spa_size) * (self.patch_size // self.spa_size)
            self.spe_seqlen = self.n_bands // self.spe_size

        else:
            self.spa_seqlen = (self.patch_size // self.spa_size + 1) * (self.patch_size // self.spa_size+1)
            self.spe_seqlen = self.n_bands // self.spe_size


        self.context_length = context_length
        self.td_model = 64
        self.embed_dim = 256
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.token_embedding = nn.Embedding(vocab_size, self.td_model)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.td_model))
        self.text_mamba1 = Mamba_Block(self.td_model, self.context_length)
        self.text_mamba2 = Mamba_Block(self.td_model, self.context_length)
        self.ln_final = LayerNorm(self.td_model)
        self.text_projection = nn.Parameter(torch.empty(self.td_model, self.embed_dim))
        self.initialize_parameters()

        self.spa1 = spa_patchfy(self.spa_size, self.patch_size, self.n_bands, self.d_model)
        self.spe1 = spe_patchfy(self.spe_size, self.patch_size, self.n_bands, self.d_model)
        self.block0 = big_mamba_block(self.layer_d_model[0], self.layer_d_model[1], self.spa_seqlen, self.spe_seqlen)
        self.block1 = big_mamba_block(self.layer_d_model[1], self.layer_d_model[2], self.spa_seqlen, self.spe_seqlen)
        self.block2 = big_mamba_block(self.layer_d_model[2], self.layer_d_model[3], self.spa_seqlen, self.spe_seqlen)
        self.block3 = big_mamba_block(self.layer_d_model[3], self.layer_d_model[4], self.spa_seqlen, self.spe_seqlen)

        self.profinal = nn.Linear(self.d_model//8, self.embed_dim)

        self.zhanping1 = ss_linear_block(self.layer_d_model[1])
        self.zhanping2 = ss_linear_block(self.layer_d_model[2])
        self.zhanping3 = ss_linear_block(self.layer_d_model[3])
        self.zhanping4 = ss_linear_block(self.layer_d_model[4], end=1)

        self.weight1 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.weight2 = nn.Parameter(torch.ones(1, requires_grad=True))
        self.weight3 = nn.Parameter(torch.ones(1, requires_grad=True))

    def _get_first_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.n_bands,
                             self.patch_size, self.patch_size))
            x = x.view(x.shape[0], -1)
            s = x.size()[1]
        return s

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.td_model ** -0.5)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.text_mamba1(x)
        x = self.text_mamba2(x)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_image(self, image):
        spa_token = self.spa1(image)
        spe_token = self.spe1(image)
        spa_token, spe_token = self.block0(spa_token, spe_token)
        spa_token, spe_token = self.block1(spa_token, spe_token)
        result2 = self.zhanping2(spa_token, spe_token)
        spa_token, spe_token = self.block2(spa_token, spe_token)
        result3 = self.zhanping3(spa_token, spe_token)
        spa_token, spe_token = self.block3(spa_token, spe_token)
        result4, pro = self.zhanping4(spa_token, spe_token)
        result = result2 * self.weight2 + result3 * self.weight3 + result4
        return pro, result


    def forward(self, image, text, label):
        img_features, image_cls = self.encode_image(image)
        if self.training:
            text_features = self.encode_text(text)
            # normalized features
            image_features = img_features / img_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()
            
            logits_per_image = logits_per_image * torch.sign(logits_per_image)
            logits_per_text = logits_per_text * torch.sign(logits_per_text)
            loss_img = F.cross_entropy(logits_per_image, label.long())
            loss_text = F.cross_entropy(logits_per_text, label.long())
            return (loss_img + loss_text)/2, image_cls/3
        else:
            return image_cls/3
