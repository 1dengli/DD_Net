import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }



class FreqSep(nn.Module):

    def __init__(self, dim, h=14, w=14):
        """
        Args:
            dim (int): 输入特征的通道数
            h (int): 特征图高度，用于初始化可学习滤波器的尺寸（实际运行时会插值）
            w (int): 特征图宽度，用于初始化可学习滤波器的尺寸
        """
        super().__init__()
        self.dim = dim

        # 高频部分占80%通道，低频部分占20%
        self.complex_weight_high = nn.Parameter(
            torch.randn(2, int(dim * 0.7), h, w, dtype=torch.float32) * 0.02
        )
        self.complex_weight_low = nn.Parameter(
            torch.randn(2, dim - int(dim * 0.7), h, w, dtype=torch.float32) * 0.02
        )
        self.prj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征，形状为 (B, C, H, W)
        Returns:
            输出特征，形状为 (B, C, H, W)
        """
        B, C, H, W = x.shape
        radius = H // 8  # 高频/低频划分半径

        # 按通道拆分（前80%为高频，后20%为低频）
        total_size = C
        split_index = int(total_size * 0.7)
        x_h, x_l = torch.split(x, [split_index, total_size - split_index], dim=1)

        # ========== 高频支路 ==========
        x_1 = torch.fft.rfft2(x_h, dim=(2, 3), norm='ortho')
        filter_size = x_1.shape[-2:]
        center_y, center_x = filter_size[0] // 2, filter_size[1] // 2
        filter_tensor = torch.ones(filter_size, device=x.device)
        y, x_coord = torch.meshgrid(
            torch.arange(filter_size[0], device=x.device),
            torch.arange(filter_size[1], device=x.device),
            indexing='ij'
        )
        distance_from_center = torch.sqrt((y - center_y) ** 2 + (x_coord - center_x) ** 2)
        filter_tensor[distance_from_center < radius] = 0  # 高通：中心抑制

        # 将可学习权重插值到当前尺寸并转为复数
        weight_high = F.interpolate(
            self.complex_weight_high, size=x_1.shape[2:4],
            mode='bilinear', align_corners=True
        ).permute(1, 2, 3, 0)  # (C, H, W, 2)
        weight_high = torch.view_as_complex(weight_high.contiguous())
        x1 = x_1 * filter_tensor * weight_high
        x1 = torch.fft.irfft2(x1, s=(H, W), dim=(2, 3), norm='ortho')

        # ========== 低频支路 ==========
        x_2 = torch.fft.rfft2(x_l, dim=(2, 3), norm='ortho')
        filter_size = x_2.shape[-2:]
        center_y, center_x = filter_size[0] // 2, filter_size[1] // 2
        filter_tensor = torch.ones(filter_size, device=x.device)
        y, x_coord = torch.meshgrid(
            torch.arange(filter_size[0], device=x.device),
            torch.arange(filter_size[1], device=x.device),
            indexing='ij'
        )
        distance_from_center = torch.sqrt((y - center_y) ** 2 + (x_coord - center_x) ** 2)
        filter_tensor[distance_from_center > radius] = 0  # 低通：只保留中心

        weight_low = F.interpolate(
            self.complex_weight_low, size=x_2.shape[2:4],
            mode='bilinear', align_corners=True
        ).permute(1, 2, 3, 0)
        weight_low = torch.view_as_complex(weight_low.contiguous())
        x2 = x_2 * filter_tensor * weight_low
        x2 = torch.fft.irfft2(x2, s=(H, W), dim=(2, 3), norm='ortho')

        # 合并并投影
        x = torch.cat((x1, x2), dim=1)
        x = self.prj(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 init_values=1e-5, feat_h=None, feat_w=None):
        """
        Args:
            feat_h, feat_w: 特征图的高度和宽度（即 img_size // patch_size）
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = FreqSep(dim, h=feat_h, w=feat_w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, spatial_size=None):
        """
        Args:
            x: 输入形状 (B, N, C)
            spatial_size: (H, W) 可选，若不提供则从 N 推断
        """
        B, N, C = x.shape
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        # ---------- 数据格式转换 (B,N,C) -> (B,C,H,W) ----------
        # 先经过 norm1
        x_norm = self.norm1(x)  # (B,N,C)
        # 重塑为 (B, C, H, W)
        x_img = x_norm.permute(0, 2, 1).view(B, C, H, W)

        # ---------- FreqSep 处理 ----------
        out_img = self.filter(x_img)  # (B, C, H, W)

        # ---------- 转换回 (B,N,C) ----------
        out_seq = out_img.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        # 残差连接 + 后续处理
        x = x + self.drop_path(self.gamma * self.norm2(out_seq + x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class fre_domain(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 计算特征图尺寸（用于 FreqSep 初始化）
        h = img_size // patch_size
        w = img_size // patch_size

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                feat_h=h, feat_w=w  # 传递特征图尺寸
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


if __name__ == "__main__":
    import torch

    model = fre_domain(
        img_size=128,
        patch_size=16,
        embed_dim=512
    )
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    print("Input shape:", x.shape)

    x1 = model(x)  # 输出形状 (1, N, C)

    B, N, C = x1.shape
    H = W = int(N ** 0.5)
    x2 = x1.permute(0, 2, 1).reshape(B, C, H, W)
    print("\nStage1 feature:", x2.shape)