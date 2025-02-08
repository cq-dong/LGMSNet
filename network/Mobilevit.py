import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Reduce
from network.Tinyunet import CMRF ,CMRFnew

# helpers




class CMUNeXtBlock_MK_resiual2(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
    def __init__(self, ch_in, ch_out,
                group_kernel_sizes= [3, 5, 7, 9],
                ): # ch_in 需要是4的倍数
        super().__init__()
        self.ch_in=ch_in
        assert self.ch_in // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.ch_in // 4
        self.group_kernel_sizes = group_kernel_sizes
        self.norm_act = nn.Sequential(nn.GroupNorm(4, ch_in),nn.GELU())

        self.local_dwc = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0], 
                padding=group_kernel_sizes[0] // 2, groups=self.group_chans)
        self.global_dwc_s = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
                padding=group_kernel_sizes[1] // 2, groups=self.group_chans)
        self.global_dwc_m = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2], 
                padding=group_kernel_sizes[2] // 2, groups=self.group_chans)
        self.global_dwc_l = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3], 
                padding=group_kernel_sizes[3] // 2, groups=self.group_chans)
        self.conv_1x1_bn=conv_1x1_bn(ch_in,self.group_chans)
        self.conv_nxn_bn=conv_nxn_bn(self.group_chans,ch_in)


    def forward(self, x):
        xclone=x.clone()
        b, c, h_, w_ = x.size()
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x, self.group_chans, dim=1)
        x_attn =self.norm_act(torch.cat((
            self.local_dwc(l_x),
            self.global_dwc_s(g_x_s),
            self.global_dwc_m(g_x_m),
            self.global_dwc_l(g_x_l),
        ), dim=1)+xclone)
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        x_attn = self.conv_1x1_bn(x_attn)
        x=self.conv_nxn_bn(x_attn)
        return x+xclone



class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

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


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    if kernel_size==2:
        return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
        )
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
class conv_nxn_bn_2(nn.Module):
    def __init__(self,inp, oup, kernel_size=2, stride=1,padding_position="left"):
        super().__init__()
        self.conv=nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.SiLU()
            )
        self.padding_position=padding_position
    def forward(self, x):
        if self.padding_position=="left":
            x_padded = F.pad(x, (0, 1, 0, 1))
        else:
            x_padded = F.pad(x, (1, 0, 1, 0))
        x=self.conv(x_padded)

        return x

class CONV_nn_bn(nn.Module):
    def __init__(self,inp, oup, kernel_size=3, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(inp, (inp+oup)//4, 1, 1, 0, bias=False)
        self.conv2=nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False)

    def forward(self, x):
        return self.conv2(x)
# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
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


## 还可以考虑，q,k 和patch如何
class Attention_with_Vconv(nn.Module):
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
        self.conv=nn.Conv2d(inner_dim,inner_dim,3,1,1)  # 考虑有或没有feature间的交融，也就是是否用DWConv
        self.height=int(math.sqrt(dim))  # TODO 偷个懒，这个输入维度h=w，所以直接开根
        self.weight=int(math.sqrt(dim))
    def conv_V(self,v):
        v=rearrange(v, 'b p h n d -> b p n (h d)').transpose(2,3)
        b,p,n,num=v.shape
        v=rearrange(v,'b p dim (height weight) -> (b p) dim height weight',b=b,p=p,height=int(math.sqrt(num)), weight=int(math.sqrt(num)))  # TODO 偷个懒，这个输入维度h=w，所以直接开根
        v=self.conv(v)
        v=rearrange(v,'(b p) dim height weight -> b p dim (height weight)',b=b,p=p).transpose(2,3)
        v=rearrange(v, 'b p n (h d)->b p h n d ', h=self.heads)
        return v


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        v=self.conv_V(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Attention_with_VDWconv(nn.Module):
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
        self.conv=nn.Conv2d(inner_dim,inner_dim,3,1,1,groups=inner_dim)  # 考虑有或没有feature间的交融，也就是是否用DWConv

    def conv_V(self,v):   # TODO: 这里的实现方式，是否合理？
        # print(v.shape)
        v=rearrange(v, 'b p h n d -> b p n (h d)').transpose(2,3)
        b,p,n,num=v.shape
        v=rearrange(v,'b p dim (height weight) -> (b p) dim height weight',b=b,p=p,height=int(math.sqrt(num)), weight=int(math.sqrt(num)))  # TODO 偷个懒，这个输入维度h=w，所以直接开根
        v=self.conv(v)
        v=rearrange(v,'(b p) dim height weight -> b p dim (height weight)',b=b,p=p).transpose(2,3)
        v=rearrange(v, 'b p n (h d)->b p h n d ', h=self.heads)
        return v


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        v=self.conv_V(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


Attention_model={
    "Attention":Attention,
    "Attention_with_Vconv":Attention_with_Vconv,
    "Attention_with_VDWconv":Attention_with_VDWconv,
}

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,attention="Attention"):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_model[attention](dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x




class Transformer_noffn(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,attention="Attention"):
        super().__init__()
        # self.layers = nn.ModuleList([])
        self.layers=nn.Sequential(*[Attention_model[attention](dim, heads, dim_head, dropout) for i in range(depth)])
        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         Attention_model[attention](dim, heads, dim_head, dropout),
        #         # FeedForward(dim, mlp_dim, dropout)
        #     ]))

    def forward(self, x):
        # for attn in self.layers:
        x = self.layers(x) + x
            # x = ff(x) + x
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
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out

class AttentionQV0(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,hw=[32,32], dropout=0.):
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
        # 后面可以考虑用（1，3）的卷积，然后padin用对侧边缘值，这就相当于是守卫连接了
        self.hw=hw
        self.reflection_padw1 = nn.ReflectionPad2d((0, 1, 0, 0))  # 左右填充1，上下填充0
        self.reflection_padw2 = nn.ReflectionPad2d((0, 0, 0, 1))  # 左右填充1，上下填充0

        self.convw1 = nn.Conv2d(hw[0]*hw[0], hw[0]*hw[0], (1, 2), 1, groups=hw[0]*hw[0], bias=False)
        self.convw2 = nn.Conv2d(hw[0]*hw[0], hw[0]*hw[0], (2, 1), 1, groups=hw[0]*hw[0], bias=False)

        self.conv_1x1=conv_1x1_bn(hw[0]*hw[0], hw[0]*hw[0])    
    
    def conv_attn(self,attn):
        attn = rearrange(attn, 'b p h (h1 w1) (h2 w2) -> b p h (h1 h2) w1 w2',h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        # TODO 这块能改的方式有很多
        attn=self.convw1(self.reflection_padw1(attn))
        attn=self.convw2(self.reflection_padw2(attn))

        attn=self.conv_1x1(attn)
        return attn


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots=self.conv_attn(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class TransformerQV0(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim,hw, dropout=0.,attention="AttentionQV1"):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionQV_model[attention](dim, heads, dim_head,hw, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class AttentionQV1_32(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,dropout=0.,hw=[int(32//2),int(32//2)] ):
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
        # 后面可以考虑用（1，3）的卷积，然后padin用对侧边缘值，这就相当于是守卫连接了
        self.hw=hw
        self.reflection_padw1 = nn.ReflectionPad2d((0, 1, 0, 0))  # 左右填充1，上下填充0
        self.reflection_padw2 = nn.ReflectionPad2d((0, 0, 0, 1))  # 左右填充1，上下填充0

        HH=16*16
        self.convw1 = nn.Conv2d(HH, HH, (1, 2), 1, groups=HH, bias=False)
        self.convw2 = nn.Conv2d(HH, HH, (2, 1), 1, groups=HH, bias=False)

        self.conv_1x1=conv_1x1_bn(HH, HH)    
    
    def conv_attn(self,attn):

        attn = rearrange(attn, 'b p h (h1 w1) (h2 w2) -> b p h (h1 h2) w1 w2',h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        # TODO 这块能改的方式有很多
        b,p,h,_,_,_=attn.shape
        attn=rearrange(attn,'b p h hh w1 w2 -> (b p h) hh w1 w2')
        print(attn.shape)
        attn=self.convw1(self.reflection_padw1(attn))
        attn=self.convw2(self.reflection_padw2(attn))

        attn=self.conv_1x1(attn)

        attn = rearrange(attn,'(b p h) (h1 h2) w1 w2 -> b p h (h1 w1) (h2 w2)',b=b,p=p,h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        return attn


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # print("1:",dots.shape)
        dots=self.conv_attn(dots)
        # print("2:",dots.shape)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class AttentionQV1_HH(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,dropout=0.,hw=[int(32//2),int(32//2)]):
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
        # 后面可以考虑用（1，3）的卷积，然后padin用对侧边缘值，这就相当于是守卫连接了
        self.hw=hw
        self.reflection_padw1 = nn.ReflectionPad2d((0, 1, 0, 0))  # 左右填充1，上下填充0
        self.reflection_padw2 = nn.ReflectionPad2d((0, 0, 0, 1))  # 左右填充1，上下填充0

        HH=hw[0]*hw[0]
        self.convw1 = nn.Conv2d(HH, HH, (1, 2), 1, groups=HH, bias=False)
        self.convw2 = nn.Conv2d(HH, HH, (2, 1), 1, groups=HH, bias=False)

        self.conv_1x1=conv_1x1_bn(HH, HH)    
    
    def conv_attn(self,attn):

        attn = rearrange(attn, 'b p h (h1 w1) (h2 w2) -> b p h (h1 h2) w1 w2',h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        # TODO 这块能改的方式有很多
        b,p,h,_,_,_=attn.shape
        attn=rearrange(attn,'b p h hh w1 w2 -> (b p h) hh w1 w2')
        # print(attn.shape)
        attn=self.convw1(self.reflection_padw1(attn))
        attn=self.convw2(self.reflection_padw2(attn))

        attn=self.conv_1x1(attn)

        attn = rearrange(attn,'(b p h) (h1 h2) w1 w2 -> b p h (h1 w1) (h2 w2)',b=b,p=p,h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        return attn


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots=self.conv_attn(dots)


        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class AttentionQV2_HH(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,dropout=0.,hw=[int(32//2),int(32//2)]):
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
        # 后面可以考虑用（1，3）的卷积，然后padin用对侧边缘值，这就相当于是守卫连接了
        self.hw=hw
        self.reflection_padw1 = nn.ReflectionPad2d((0, 1, 0, 0))  # 左右填充1，上下填充0
        self.reflection_padw2 = nn.ReflectionPad2d((0, 0, 0, 1))  # 左右填充1，上下填充0

        HH=hw[0]*hw[0]
        self.convw1 = nn.Conv2d(HH, HH, (1, 2), 1, groups=HH, bias=False)
        self.convw2 = nn.Conv2d(HH, HH, (2, 1), 1, groups=HH, bias=False)

        self.conv_1x1=conv_1x1_bn(HH, HH)    
    
    def conv_attn(self,attn):

        attn = rearrange(attn, 'b p h (h1 w1) (h2 w2) -> b p h (h1 h2) w1 w2',h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        # TODO 这块能改的方式有很多
        b,p,h,_,_,_=attn.shape
        attn=rearrange(attn,'b p h hh w1 w2 -> (b p h) hh w1 w2')
        # print(attn.shape)
        attn1=self.convw1(self.reflection_padw1(attn))
        attn2=self.convw2(self.reflection_padw2(attn))

        attn=self.conv_1x1(attn1+attn2)

        attn = rearrange(attn,'(b p h) (h1 h2) w1 w2 -> b p h (h1 w1) (h2 w2)',b=b,p=p,h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        return attn


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots=self.conv_attn(dots)


        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class AttentionQV3_HH(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,dropout=0.,hw=[int(32//2),int(32//2)]):
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
        # 后面可以考虑用（1，3）的卷积，然后padin用对侧边缘值，这就相当于是守卫连接了
        self.hw=hw
        self.reflection_padw1 = nn.ReflectionPad2d((0, 1, 0, 0))  # 左右填充1，上下填充0
        self.reflection_padw2 = nn.ReflectionPad2d((0, 0, 0, 1))  # 左右填充1，上下填充0

        HH=hw[0]*hw[0]
        self.convw1 = nn.Conv2d(HH, HH, (1, 3), 1, groups=HH, bias=False)
        self.convw2 = nn.Conv2d(HH, HH, (3, 1), 1, groups=HH, bias=False)

        self.conv_1x1=conv_1x1_bn(HH, HH)    
    def pad_tensor_H(self,tensor):   # 在倒数第二个维度pad
        B, C, H, W = tensor.size()
        # 提取 dim 维度的最后一个元素和第一个元素
        last_column = tensor[:, :, -1, :]  # 形状为 [B, C, 1, H]
        first_column = tensor[:, :, 0, :]  # 形状为 [B, C, 1, H]
        # 在 dim 维度进行填充
        padded_tensor = F.pad(tensor, (0,0,1, 1), mode='constant', value=0)  # 先在 W 维度上扩展一个位置
        padded_tensor[:, :, 0, :] = last_column  # 上方填充 W[-1]
        padded_tensor[:, :, -1,:] = first_column  # 下方填充 W[0]
        return padded_tensor
    def pad_tensor_W(self,tensor):   # 在倒数第1个维度pad
        B, C, H, W = tensor.size()
        # 提取 dim 维度的最后一个元素和第一个元素
        last_column = tensor[:, :, :, -1]  # 形状为 [B, C, H, 1]
        first_column = tensor[:, :, :, 0]  # 形状为 [B, C, H, 1]
        # 在 dim 维度进行填充
        padded_tensor = F.pad(tensor, (1, 1), mode='constant', value=0)  # 先在 W 维度上扩展一个位置
        padded_tensor[:, :, :, 0] = last_column  # 上方填充 W[-1]
        padded_tensor[:, :, :, -1] = first_column  # 下方填充 W[0]

        return padded_tensor

    def conv_attn(self,attn):

        attn = rearrange(attn, 'b p h (h1 w1) (h2 w2) -> b p h (h1 h2) w1 w2',h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        # TODO 这块能改的方式有很多
        b,p,h,_,_,_=attn.shape
        attn=rearrange(attn,'b p h hh w1 w2 -> (b p h) hh w1 w2')
        # print(attn.shape)
        attn1=self.convw1(self.pad_tensor_W(attn))
        attn2=self.convw2(self.pad_tensor_H(attn))

        attn=self.conv_1x1(attn1+attn2)

        attn = rearrange(attn,'(b p h) (h1 h2) w1 w2 -> b p h (h1 w1) (h2 w2)',b=b,p=p,h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        return attn


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots=self.conv_attn(dots)


        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class AttentionQV4_HH(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,dropout=0.,hw=[int(32//2),int(32//2)]):
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
        # 后面可以考虑用（1，3）的卷积，然后padin用对侧边缘值，这就相当于是守卫连接了
        self.hw=hw
        self.reflection_padw1 = nn.ReflectionPad2d((0, 1, 0, 0))  # 左右填充1，上下填充0
        self.reflection_padw2 = nn.ReflectionPad2d((0, 0, 0, 1))  # 左右填充1，上下填充0

        HH=hw[0]*hw[0]
        self.convw1 = nn.Conv2d(HH, HH, (1, 3), 1, groups=HH, bias=False)
        self.convw2 = nn.Conv2d(HH, HH, (3, 1), 1, groups=HH, bias=False)

        self.conv_1x1=conv_1x1_bn(HH, HH)    
    def pad_tensor_H(self,tensor):   # 在倒数第二个维度pad
        B, C, H, W = tensor.size()
        # 提取 dim 维度的最后一个元素和第一个元素
        last_column = tensor[:, :, -1, :]  # 形状为 [B, C, 1, H]
        first_column = tensor[:, :, 0, :]  # 形状为 [B, C, 1, H]
        # 在 dim 维度进行填充
        padded_tensor = F.pad(tensor, (0,0,1, 1), mode='constant', value=0)  # 先在 W 维度上扩展一个位置
        padded_tensor[:, :, 0, :] = last_column  # 上方填充 W[-1]
        padded_tensor[:, :, -1,:] = first_column  # 下方填充 W[0]
        return padded_tensor
    def pad_tensor_W(self,tensor):   # 在倒数第1个维度pad
        B, C, H, W = tensor.size()
        # 提取 dim 维度的最后一个元素和第一个元素
        last_column = tensor[:, :, :, -1]  # 形状为 [B, C, H, 1]
        first_column = tensor[:, :, :, 0]  # 形状为 [B, C, H, 1]
        # 在 dim 维度进行填充
        padded_tensor = F.pad(tensor, (1, 1), mode='constant', value=0)  # 先在 W 维度上扩展一个位置
        padded_tensor[:, :, :, 0] = last_column  # 上方填充 W[-1]
        padded_tensor[:, :, :, -1] = first_column  # 下方填充 W[0]

        return padded_tensor

    def conv_attn(self,attn):

        attn = rearrange(attn, 'b p h (h1 w1) (h2 w2) -> b p h (h1 h2) w1 w2',h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        # TODO 这块能改的方式有很多
        b,p,h,_,_,_=attn.shape
        attn=rearrange(attn,'b p h hh w1 w2 -> (b p h) hh w1 w2')
        # print(attn.shape)
        attn1=self.convw1(self.pad_tensor_W(attn))
        attn2=self.convw2(self.pad_tensor_H(attn1))

        attn=self.conv_1x1(attn1+attn2)

        attn = rearrange(attn,'(b p h) (h1 h2) w1 w2 -> b p h (h1 w1) (h2 w2)',b=b,p=p,h1=self.hw[0], w1=self.hw[1], h2=self.hw[0], w2=self.hw[1])
        return attn


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots=self.conv_attn(dots)


        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


AttentionQV_model_HH={
    "AttentionQV1_HH":AttentionQV1_HH,
    "AttentionQV2_HH":AttentionQV2_HH,
    "AttentionQV3_HH":AttentionQV3_HH,
    "AttentionQV4_HH":AttentionQV4_HH,
    }

class TransformerQV_HH(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim,dropout=0.,attention="AttentionQV1_HH",hw=[int(32//2),int(32//2)]):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionQV_model_HH[attention](dim, heads, dim_head,dropout,hw=hw),
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileViTBlockQV_HH(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="AttentionQV1_HH",hw=[int(32//2),int(32//2)]):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = TransformerQV_HH(dim, depth, 4, 8, mlp_dim, dropout,attention=attention,hw=hw)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        # print("x.shape",x.shape,"x.shape")

        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # print("x.shape",x.shape,"x.shape1")
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention"):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout,attention=attention)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        # print("x.shape",x.shape,"x.shape")

        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # print("x.shape",x.shape,"x.shape1")
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class MobileViTBlock_PE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention",pos_len=32//2*32//2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.pos_embedding = nn.Parameter(torch.randn(1,1, pos_len, dim))

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout,attention=attention)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)




    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        # print("x.shape",x.shape,"x.shape")

        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # print("x.shape",x.shape,"x.shape1")
        x = self.transformer(x+self.pos_embedding)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self,device):
        # self.encoding


        return self.encoding.to(device)
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class MobileViTBlock_SCPE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention",pos_len=32//2*32//2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.pos_embedding = PositionalEncoding(dim, pos_len)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout,attention=attention)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)



    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        # print("x.shape",x.shape,"x.shape")

        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # print("x.shape",x.shape,"x.shape1")
        x = self.transformer(x+self.pos_embedding(x.device))        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x





class MobileViTBlockQV_16(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention=1):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = TransformerQV_16(dim, depth, 4, 8, mlp_dim, dropout,attention=attention)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        # print("x.shape",x.shape,"x.shape")

        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # print("x.shape",x.shape,"x.shape1")
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class MobileViTBlockQV_32(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention=1):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = TransformerQV_32(dim, depth, 4, 8, mlp_dim, dropout,attention=attention)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        # print("x.shape",x.shape,"x.shape")

        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        # print("x.shape",x.shape,"x.shape1")
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class TinyMobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        # self.conv2 = conv_1x1_bn(channel, dim)
        self.conv1 = CMRF(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        # self.conv3 = conv_1x1_bn(dim, channel)
        self.conv3 = CMRF(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)



    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
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



class TinyMobileViTBlock_v2(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        # self.conv2 = conv_1x1_bn(channel, dim)
        self.conv1 = CMRFnew(dim, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)

        # self.conv3 = conv_1x1_bn(dim, channel)
        self.conv3 = CMRFnew(channel, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)



    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
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



class TinyMobileViTBlock_v1(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        # self.conv1 = CMRF(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        # self.conv3 = conv_1x1_bn(dim, channel)
        self.conv3 = CMRF(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)



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




class MobileViTBlocktinytem(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        # self.conv2 = conv_1x1_bn(channel, channel)
        self.conv1 = CMRFnew(dim, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+ channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
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




class MobileViTBlocktinytem2(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        # self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        # self.conv2 = conv_1x1_bn(channel, channel)
        self.conv1 = CMRFnew(dim, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)

        # self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+ channel, channel, kernel_size)
        self.conv3 = CMRFnew(channel, channel)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
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





class MobileViTBlocktem_noffn(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer_noffn(channel, depth, 4, 8, mlp_dim, dropout)

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



def apply_random_mask(x, mask_ratio_N, mask_ratio_F):
    # 获取输入张量的设备
    device = x.device
    
    # 获取输入张量的大小
    b, c, N, F = x.shape
    
    # 在N维度上生成随机mask，并将其移动到与x相同的设备
    mask_N = (torch.rand(b, c, N, device=device) > mask_ratio_N).float()  # 维度 N 上的mask
    
    # 在F维度上生成随机mask，并将其移动到与x相同的设备
    mask_F = (torch.rand(b, c, F, device=device) > mask_ratio_F).float()  # 维度 F 上的mask

    # 对输入张量分别应用N和F的mask
    x_masked = x * mask_N.unsqueeze(-1) * mask_F.unsqueeze(-2)
    
    return x_masked

class MobileViTBlocktem_mask(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,mask=[0,0]):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+ channel, channel, kernel_size)
        self.mask = mask

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = apply_random_mask(x, mask_ratio_N=self.mask[0], mask_ratio_F=self.mask[1])
        x = self.transformer(x)        
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlocktem_CT(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, spilt_list=[96,32],dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.conv_spilt = DWConv2d_BN_ReLU(spilt_list[1],spilt_list[1],3)
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
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x1 = self.transformer(x1)        
        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = torch.cat((x1, x2), 1)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlocktem_CT_3(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, spilt_list=[96,32],dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.conv_spilt = CMUNeXtBlock_MK_resiual2(spilt_list[1],spilt_list[1])
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
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x1 = self.transformer(x1)        
        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = torch.cat((x1, x2), 1)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlocktem_CT_2(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, spilt_list=[96,32],dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn_2(dim, channel, kernel_size=2,padding_position="left")
        self.conv2 = conv_1x1_bn(channel, channel)

        self.conv_spilt = DWConv2d_BN_ReLU(spilt_list[1],spilt_list[1],3)
        self.transformer = Transformer(spilt_list[0], depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn_2(dim+ channel, channel, kernel_size=2,padding_position="right")
        self.spilt_list=spilt_list

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        x1,x2 = torch.split(x,self.spilt_list, dim=1)
        x2 = self.conv_spilt(x2)+x2


        # Global representations
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x1 = self.transformer(x1)        
        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = torch.cat((x1, x2), 1)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlocktem_CT_mask(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, spilt_list=[96,32],dropout=0.,mask=[0,0]):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.conv_spilt = DWConv2d_BN_ReLU(spilt_list[1],spilt_list[1],3)
        self.transformer = Transformer(spilt_list[0], depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+ channel, channel, kernel_size)
        self.spilt_list=spilt_list
        self.mask = mask


    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        x1,x2 = torch.split(x,self.spilt_list, dim=1)
        x2 = self.conv_spilt(x2)+x2


        # Global representations
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x1 = apply_random_mask(x1, mask_ratio_N=self.mask[0], mask_ratio_F=self.mask[1])

        x1 = self.transformer(x1)        
        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x = torch.cat((x1, x2), 1)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlocktem_CIT(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, spilt_list=[48,48,32,32//2*32//2],dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.conv_spilt = DWConv2d_BN_ReLU(spilt_list[2],spilt_list[2],3)
        self.transformer1 = Transformer(spilt_list[0], depth, 4, 8, mlp_dim, dropout)
        self.transformer2 = Transformer(spilt_list[3], depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(channel, channel)
        self.conv4 = conv_nxn_bn(dim+ channel, channel, kernel_size)
        self.spilt_list=spilt_list

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        x1,x2,x3 = torch.split(x,self.spilt_list[:3], dim=1)
        x3 = self.conv_spilt(x3)+x3


        # Global representations
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x1 = self.transformer1(x1)        
        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        _, _, h, w = x2.shape
        x2 = rearrange(x2, 'b d (h ph) (w pw) -> b (ph pw) d (h w)', ph=self.ph, pw=self.pw)
        x2 = self.transformer2(x2)        
        x2 = rearrange(x2, 'b (ph pw) d (h w) -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        x = torch.cat((x1,x2,x3), 1)
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class MobileViTBlocktemtem(nn.Module):
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
        x = self.transformer(x)      
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class MobileViTBlocktemnew(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer1 = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        self.transformer2 = Transformer(channel, depth, 4, 8, mlp_dim, dropout)

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
        x = self.transformer1(x)  
        x = self.transformer2(x)      
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class MobileViTBlocktemSCPE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention",pos_len=32//2*32//2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        self.pos_embedding = PositionalEncoding(channel, pos_len)

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
        x = self.transformer(x+self.pos_embedding(x.device))        
        # x = self.transformer(x)      不小心多了一行，重新运行去重  
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlocktem1SCPE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention",pos_len=32//2*32//2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        self.pos_embedding = PositionalEncoding(channel, pos_len)

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
        x = self.transformer(x+self.pos_embedding(x.device))        
        x = self.transformer(x)     # 不小心多了一行，重新运行去重  
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViTBlocktem2SCPE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention",pos_len=32//2*32//2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        self.pos_embedding = PositionalEncoding(channel, pos_len)

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
        x = self.transformer(x+self.pos_embedding(x.device))        
        x = self.transformer(x+self.pos_embedding(x.device))     # 不小心多了一行，重新运行去重  
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlocktem3SCPE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention",pos_len=32//2*32//2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        self.pos_embedding = PositionalEncoding(channel, pos_len)

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
        x = self.transformer(x+self.pos_embedding(x.device))        
        x = self.transformer(x+self.pos_embedding(x.device))     # 不小心多了一行，重新运行去重  
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViTBlocktem4SCPE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention",pos_len=32//2*32//2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer1 = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        self.transformer2 = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        self.pos_embedding = PositionalEncoding(channel, pos_len)

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
        x = self.transformer1(x+self.pos_embedding(x.device))        
        x = self.transformer2(x+self.pos_embedding(x.device))     # 不小心多了一行，重新运行去重  

        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class MobileViTBlocktemtemSCPE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="Attention",pos_len=32//2*32//2):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = Transformer(channel, depth, 4, 8, mlp_dim, dropout)
        self.pos_embedding = PositionalEncoding(channel, pos_len)

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
        x = self.transformer(x+self.pos_embedding(x.device))        
        x = self.transformer(x)      #不小心多了一行，重新运行去重  
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x




class MobileViTBlockQVtemSCPE(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.,attention="AttentionQV1_HH",pos_len=32//2*32//2,hw=[int(32//2),int(32//2)]):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(dim, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, channel)

        self.transformer = TransformerQV_HH(channel, depth, 4, 8, mlp_dim, dropout,attention=attention,hw=hw)
        self.pos_embedding = PositionalEncoding(channel, pos_len)

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
        # print(x.shape)
        x = self.transformer(x+self.pos_embedding(x.device))        
        # x = self.transformer(x)      不小心多了一行，重新运行去重  
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
