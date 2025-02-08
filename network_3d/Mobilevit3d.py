import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Reduce
import numpy as np
# helpers

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.InstanceNorm3d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.InstanceNorm3d(oup),
        nn.SiLU()
    )

# classes

class FeedForward3D(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),  # 1D convolution with kernel size 1  #### ?????? TODO
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),  # 1D convolution with kernel size 1
            nn.Dropout(dropout)

        )

    def forward(self, x):
        # x is expected to be of shape (batch_size, channels, length)
        # print("x/shape",x.shape)
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward3D(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        # print("x91:",x.shape)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.InstanceNorm3d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.InstanceNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.InstanceNorm3d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.InstanceNorm3d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.InstanceNorm3d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out


def get_3d_positional_encoding(D, H, W, d_model):
    """
    D: Depth (Z axis)
    H: Height (Y axis)
    W: Width (X axis)
    d_model: The dimension of the model (usually the hidden dimension of the transformer)
    
    Returns:
    pos_encoding: Tensor of shape (D * H * W, d_model)
    """
    
    # 创建D, H, W的坐标网格
    d_idx = np.arange(D)
    h_idx = np.arange(H)
    w_idx = np.arange(W)
    
    # 使用np.meshgrid生成网格
    D_grid, H_grid, W_grid = np.meshgrid(d_idx, h_idx, w_idx, indexing='ij')
    
    # 将D, H, W的坐标展平
    pos = np.stack([D_grid.flatten(), H_grid.flatten(), W_grid.flatten()], axis=-1)
    
    # 初始化位置编码矩阵
    pos_encoding = np.zeros((D * H * W, d_model))
    
    # 对每个维度 (D, H, W) 使用正弦和余弦函数生成位置编码
    for i in range(d_model // 3):  # 将d_model分配给每个坐标轴
        for axis in range(3):  # 对三个维度 (D, H, W) 分别编码
            pos_encoding[:, i + axis * (d_model // 3)] = np.sin(pos[:, axis] / (10000 ** (i / (d_model // 3))))
            pos_encoding[:, i + axis * (d_model // 3) + 1] = np.cos(pos[:, axis] / (10000 ** (i / (d_model // 3))))
    pos_encoding=torch.tensor(pos_encoding, dtype=torch.float32)
    pos_encoding=pos_encoding.unsqueeze(0).unsqueeze(0)
    return pos_encoding

class MobileViTBlock3D(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw,self.pf = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+channel, channel, kernel_size)

    def forward(self, x):
        # print(x.shape)
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w, f = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) (f pf)-> b (ph pw pf) (h w f) d', ph=self.ph, pw=self.pw,pf=self.pf)
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw pf) (h w f) d -> b d (h ph) (w pw) (f pf)', h=h//self.ph, w=w//self.pw,f=f//self.pf, ph=self.ph, pw=self.pw,pf=self.pf)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlock3DPOS(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw,self.pf = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+channel, channel, kernel_size)

    def forward(self, x):
        # print(x.shape)
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, d, h, w, f = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) (f pf)-> b (ph pw pf) (h w f) d', ph=self.ph, pw=self.pw,pf=self.pf)
        pos_encoding=get_3d_positional_encoding(h//self.ph,w//self.pw,f//self.pf,d)
        x = self.transformer(x+pos_encoding.to(x.device))        
        x = rearrange(x, 'b (ph pw pf) (h w f) d -> b d (h ph) (w pw) (f pf)', h=h//self.ph, w=w//self.pw,f=f//self.pf, ph=self.ph, pw=self.pw,pf=self.pf)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class MobileViTBlocktem(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+ channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViTBlock1(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        # self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        # x = self.conv1(x)
        # x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViTBlock2(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        # self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        # self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        # x = self.conv1(x)
        # x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        # x = torch.cat((x, y), 1)
        # x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(dims[1], depths[1], channels[7],
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], depths[2], channels[9],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1_bn(channels[-2], last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)




class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.InstanceNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.InstanceNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model




class DWConv3d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        # Depthwise 3x3x3 convolution
        self.add_module('dwconv3x3x3',
                        nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        # Depthwise 1x1x1 convolution
        self.add_module('dwconv1x1x1',
                        nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm3d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3x3 and bn1
        dwconv3x3x3, bn1, relu, dwconv1x1x1, bn2 = self._modules.values()

        # Fuse dwconv3x3x3 and bn1
        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3x3.weight * w1[:, None, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3x3 = nn.Conv3d(w1.size(1) * dwconv3x3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3x3.stride,
                                      padding=dwconv3x3x3.padding, dilation=dwconv3x3x3.dilation, groups=dwconv3x3x3.groups,
                                      device=dwconv3x3x3.weight.device)
        fused_dwconv3x3x3.weight.data.copy_(w1)
        fused_dwconv3x3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1x1.weight * w2[:, None, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1x1 = nn.Conv3d(w2.size(1) * dwconv1x1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1x1.stride,
                                      padding=dwconv1x1x1.padding, dilation=dwconv1x1x1.dilation, groups=dwconv1x1x1.groups,
                                      device=dwconv1x1x1.weight.device)
        fused_dwconv1x1x1.weight.data.copy_(w2)
        fused_dwconv1x1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3x3, relu, fused_dwconv1x1x1)
        return fused_model



class MobileViTBlocktem_CT3D(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, spilt_list=[96,32],dropout=0.):
        super().__init__()
        self.ph, self.pw,self.pf= patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.conv_spilt = DWConv3d_BN_ReLU(spilt_list[1],spilt_list[1],3)
        self.transformer = Transformer(spilt_list[0], depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+ channel, channel, kernel_size)
        self.spilt_list=spilt_list

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        x1,x2 = torch.split(x,self.spilt_list, dim=1)
        x2 = self.conv_spilt(x2)+x2


        # Global representations
        _, _, h, w ,f= x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) (f pf)-> b (ph pw pf) (h w f) d', ph=self.ph, pw=self.pw,pf=self.pf)
        x1 = self.transformer(x1)        
        x1 = rearrange(x1, 'b (ph pw pf) (h w f) d -> b d (h ph) (w pw) (f pf)', h=h//self.ph, w=w//self.pw,f=f//self.pf, ph=self.ph, pw=self.pw,pf=self.pf)
        x = torch.cat((x1, x2), 1)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x






class MobileViT_seg(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size=96,
        dims=3,
        channels=[],
        num_classes=2,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3)
    ):
        super().__init__()
        assert len(dims) == 3, 'dims must be a tuple of 3'
        assert len(depths) == 3, 'depths must be a tuple of 3'

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion),
            MobileViTBlock(dims[0], depths[0], channels[5],
                           kernel_size, patch_size, int(dims[0] * 2))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion),
            MobileViTBlock(dims[1], depths[1], channels[7],
                           kernel_size, patch_size, int(dims[1] * 4))
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion),
            MobileViTBlock(dims[2], depths[2], channels[9],
                           kernel_size, patch_size, int(dims[2] * 4))
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1_bn(channels[-2], last_dim),
            # Reduce('b c h w -> b c', 'mean'),
            # nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)
    


if __name__ == '__main__':
    img = torch.randn(5, 3, 256, 256)

    vit = MobileViT()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))
