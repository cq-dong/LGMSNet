import torch
import torch.nn as nn
# from network.SCSA import SCSA,SCSA1
from network.CBAM import CBAM
from network.CNNVIT import CNNTblock
from network.Mobilevit import MobileViTBlock,MobileViTBlocktinytem2,MobileViTBlocktinytem,MobileViTBlocktemSCPE,MobileViTBlockQVtemSCPE,MobileViTBlocktemtemSCPE,MobileViTBlocktemnew,MobileViTBlocktemtem
from network.Mobilevit import MobileViTBlock,MobileViTBlock1,MobileViTBlock2,MobileViTBlocktem,TinyMobileViTBlock,TinyMobileViTBlock_v1,MobileViTBlocktem,MobileViTBlocktem_mask
from network.Mobilevit import MobileViTBlocktem1SCPE,MobileViTBlocktem2SCPE,MobileViTBlocktem3SCPE,MobileViTBlocktem4SCPE,MobileViTBlocktem_CT
from network.Mobilevit import MobileViTBlocktem_noffn,MobileViTBlocktem_CIT,MobileViTBlocktem_CT_mask

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class CMUNeXtBlock_MK(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
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
        b, c, h_, w_ = x.size()
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x, self.group_chans, dim=1)
        x_attn =self.norm_act(torch.cat((
            self.local_dwc(l_x),
            self.global_dwc_s(g_x_s),
            self.global_dwc_m(g_x_m),
            self.global_dwc_l(g_x_l),
        ), dim=1))
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        x_attn = self.conv_1x1_bn(x_attn)
        x=self.conv_nxn_bn(x_attn)
        return x



class CMUNeXtBlock_MK_resiual(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
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
        ), dim=1))
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        x_attn = self.conv_1x1_bn(x_attn)
        x=self.conv_nxn_bn(x_attn)
        return x+xclone




class CMUNeXtBlock_MK_resiual1(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
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
        return x




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



class CMUNeXtBlock_MK1_resiual(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
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
        self.conv_1x1_bn=conv_1x1_bn(ch_in,ch_in)
        self.conv_nxn_bn=conv_nxn_bn(ch_in,ch_in)


    def forward(self, x):
        xclone=x.clone()
        b, c, h_, w_ = x.size()
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x, self.group_chans, dim=1)
        x_attn =self.norm_act(torch.cat((
            self.local_dwc(l_x),
            self.global_dwc_s(g_x_s),
            self.global_dwc_m(g_x_m),
            self.global_dwc_l(g_x_l),
        ), dim=1))
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        x_attn = self.conv_1x1_bn(x_attn)
        x=self.conv_nxn_bn(x_attn)
        return x+xclone


class CMUNeXtBlock_MK2_resiual(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
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
        self.conv_1x1_bn=conv_1x1_bn(ch_in+self.group_chans,ch_in)
        self.conv_nxn_bn=conv_nxn_bn(ch_in,self.group_chans)

        # self.batchnorm_act=nn.Sequential(nn.BatchNorm2d(ch_in),nn.GELU())


    def forward(self, x):
        xclone=x.clone()
        b, c, h_, w_ = x.size()
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x, self.group_chans, dim=1)
        x_attn =self.norm_act(torch.cat((
            self.local_dwc(l_x),
            self.global_dwc_s(g_x_s),
            self.global_dwc_m(g_x_m),
            self.global_dwc_l(g_x_l),
        ), dim=1))
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        # x_attn = self.conv_nxn_bn(x_attn)
        x=self.conv_1x1_bn(torch.cat((x_attn,self.conv_nxn_bn(x_attn)),dim=1))
        return x+xclone


class CMUNeXtBlock_MK2_resiual2(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
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
        self.conv_1x1_bn=conv_1x1_bn(ch_in+self.group_chans,ch_in)
        self.conv_nxn_bn=conv_nxn_bn(ch_in,self.group_chans)

        # self.batchnorm_act=nn.Sequential(nn.BatchNorm2d(ch_in),nn.GELU())


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
        # x_attn = self.conv_nxn_bn(x_attn)
        x=self.conv_1x1_bn(torch.cat((x_attn,self.conv_nxn_bn(x_attn)),dim=1))
        return x+xclone



class CMUNeXtBlock_MMK(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
    def __init__(self, ch_in, ch_out,
                group_kernel_sizes= [3, 5, 7, 9],
                ): # ch_in 需要是4的倍数
        super().__init__()
        self.ch_in=ch_in
        assert self.ch_in // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.ch_in // 4
        self.group_kernel_sizes = group_kernel_sizes
        self.norm_act = nn.Sequential(nn.GroupNorm(4, ch_in),nn.GELU())

        self.local_dwc = nn.ModuleList([nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0], 
                padding=group_kernel_sizes[0] // 2, groups=self.group_chans) for i in range(4)])
        self.global_dwc_s = nn.ModuleList([nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
                padding=group_kernel_sizes[1] // 2, groups=self.group_chans) for i in range(3) ])
        self.global_dwc_m = nn.ModuleList([nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2], 
                padding=group_kernel_sizes[2] // 2, groups=self.group_chans) for i in range(2)])
        self.global_dwc_l = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3], 
                padding=group_kernel_sizes[3] // 2, groups=self.group_chans)
        
        self.BatchNorm2d=nn.ModuleList([nn.BatchNorm2d(self.group_chans) for i in range(4)])

        self.conv_1x1_bn=conv_1x1_bn(ch_in,self.group_chans)
        self.conv_nxn_bn=conv_nxn_bn(self.group_chans,ch_in)


    def forward(self, x):
        b, c, h_, w_ = x.size()
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x, self.group_chans, dim=1)
        # 注意需要考虑归一化和数值范围
        x_attn =self.norm_act(torch.cat((
            self.BatchNorm2d[0](self.local_dwc[0](l_x)),
            self.BatchNorm2d[1](self.local_dwc[1](g_x_s)+self.global_dwc_s[0](g_x_s)),
            self.BatchNorm2d[2](self.local_dwc[2](g_x_m)+self.global_dwc_s[1](g_x_m)+self.global_dwc_m[0](g_x_m)),
            self.BatchNorm2d[3](self.local_dwc[3](g_x_l)+self.global_dwc_s[2](g_x_l)+self.global_dwc_m[1](g_x_l)+self.global_dwc_l(g_x_l)),
        ), dim=1))
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        x_attn = self.conv_1x1_bn(x_attn)
        x=self.conv_nxn_bn(x_attn)
        return x




class CMUNeXtBlock_MKUP(nn.Module):  # 根据SCSA灵感，直接在一个block中为不同的channel设置不同的卷积核，同时引入通道注意力
    def __init__(self, ch_in, ch_out,
                group_kernel_sizes= [3, 5, 7, 9],
                ): # ch_in 需要是4的倍数
        super().__init__()
        self.ch_in=ch_in
        assert self.ch_in // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.ch_in 
        self.group_kernel_sizes = group_kernel_sizes
        self.norm_act = nn.Sequential(nn.GroupNorm(4, ch_in*4),nn.GELU())

        self.local_dwc = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0], 
                padding=group_kernel_sizes[0] // 2, groups=self.group_chans)
        self.global_dwc_s = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
                padding=group_kernel_sizes[1] // 2, groups=self.group_chans)
        self.global_dwc_m = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2], 
                padding=group_kernel_sizes[2] // 2, groups=self.group_chans)
        self.global_dwc_l = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3], 
                padding=group_kernel_sizes[3] // 2, groups=self.group_chans)
        self.conv_1x1_bn=conv_1x1_bn(ch_in*4,self.group_chans)
        self.conv_nxn_bn=conv_nxn_bn(self.group_chans,ch_out)


    def forward(self, x):
        b, c, h_, w_ = x.size()
        l_x=self.local_dwc(x)
        g_x_s=self.global_dwc_s(l_x) 
        g_x_m=self.global_dwc_m(g_x_s)
        g_x_l = self.global_dwc_l(g_x_m)
        # print(g_x_l.shape,g_x_m.shape,g_x_l.shape,l_x.shape)
        x_attn =self.norm_act(torch.cat((
            l_x,
            g_x_s,
            g_x_m,
            g_x_l,
        ), dim=1))
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        x_attn = self.conv_1x1_bn(x_attn)
        x=self.conv_nxn_bn(x_attn)
        return x



# 直接把 block改为
class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CMUNeXt(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_MK(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1





class CMUNeXt_MK_VIT_1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_MK_VIT_1_v1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)),conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)),conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_MK_resiual(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


Block_dict={
    "CMUNeXtBlock_MK_resiual":CMUNeXtBlock_MK_resiual,
    "CMUNeXtBlock_MK_resiual1":CMUNeXtBlock_MK_resiual1,
    "CMUNeXtBlock_MK_resiual2":CMUNeXtBlock_MK_resiual2,
    "CMUNeXtBlock_MK1_resiual":CMUNeXtBlock_MK1_resiual,
    "CMUNeXtBlock_MK2_resiual":CMUNeXtBlock_MK2_resiual,
    "CMUNeXtBlock_MK2_resiual2":CMUNeXtBlock_MK2_resiual2,
}

class CMUNeXt_MK_resiual_model(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class DU_MK_resiual_model(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(conv_1x1_bn(dims[3]*2,dims[3]*2),CMUNeXtBlockmodel(ch_in=dims[3]*2, ch_out=dims[3]*2),conv_1x1_bn(dims[3]*2,dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(conv_1x1_bn(dims[2]*2,dims[2]*2),CMUNeXtBlockmodel(ch_in=dims[2]*2, ch_out=dims[2]*2),conv_1x1_bn(dims[2]*2,dims[2]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = nn.Sequential(conv_1x1_bn(dims[1]*2,dims[1]*2),CMUNeXtBlockmodel(ch_in=dims[1]*2, ch_out=dims[1]*2),conv_1x1_bn(dims[1]*2,dims[1]))
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = nn.Sequential(conv_1x1_bn(dims[0]*2,dims[0]*2),CMUNeXtBlockmodel(ch_in=dims[0]*2, ch_out=dims[0]*2),conv_1x1_bn(dims[0]*2,dims[0]))
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_MK_resiual_model_1(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(MobileViTBlocktem(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvit1(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(MobileViTBlocktinytem(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktinytem(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1

class MKtinyvit3(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktinytem2(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktinytem2(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktinytem2(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1

class MKtinyvit2(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(MobileViTBlocktem(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class MKtinyvit4(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvit4_noffn(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],atten_depth=[1,1,1,1]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem_noffn(dims[2], atten_depth[0],dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem_noffn(dims[3], atten_depth[1],dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem_noffn(dims[3]* 2, atten_depth[2],dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem_noffn(dims[2], atten_depth[3],dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class MKtinyvit4_attendepth(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],atten_depth=[1,1,1,1]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem(dims[2], atten_depth[0],dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem(dims[3], atten_depth[1],dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem(dims[3]* 2, atten_depth[2],dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem(dims[2], atten_depth[3],dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class MKtinyvit4_mask(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],mask=[[0,0],[0,0],[0,0],[0,0]]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem_mask(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),mask=[mask[0][0],mask[0][1]]))
        self.encoder5 = nn.Sequential(MobileViTBlocktem_mask(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2),mask=[mask[1][0],mask[1][1]]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem_mask(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),mask=[mask[2][0],mask[2][1]])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem_mask(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2),mask=[mask[3][0],mask[3][1]]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvit4_2(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem(dims[2], 1,dims[3],kernel_size=3, patch_size=(1,1), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem(dims[3], 1,dims[4],kernel_size=3, patch_size=(1,1), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(1,1), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class MKtinyvit4_3(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem(dims[2], 1,dims[3],kernel_size=3, patch_size=(1,1), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem(dims[3], 1,dims[4],kernel_size=3, patch_size=(1,1), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(1,1), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem(dims[2], 1,dims[2],kernel_size=3, patch_size=(1,1), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvit4_1(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class MKtinyvit4_CT(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",spilt_list=[[96,32],[96,32],[96,32],[48,16]],input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem_CT(dims[2], 1,dims[3],kernel_size=3,spilt_list=spilt_list[0], patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem_CT(dims[3], 1,dims[4],kernel_size=3,spilt_list=spilt_list[1],  patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem_CT(dims[3]* 2, 1,dims[3],kernel_size=3, spilt_list=spilt_list[2], patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem_CT(dims[2], 1,dims[2],kernel_size=3,spilt_list=spilt_list[3], patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class MKtinyvit4_CTmask(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",spilt_list=[[96,32],[96,32],[96,32],[48,16]],input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],mask=[[0,0],[0,0],[0,0],[0,0]]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem_CT_mask(dims[2], 1,dims[3],kernel_size=3,spilt_list=spilt_list[0], patch_size=(2,2), mlp_dim=int(dims[3] * 2),mask=[mask[0][0],mask[0][1]]))
        self.encoder5 = nn.Sequential(MobileViTBlocktem_CT_mask(dims[3], 1,dims[4],kernel_size=3,spilt_list=spilt_list[1],  patch_size=(2,2), mlp_dim=int(dims[4] * 2),mask=[mask[0][0],mask[1][1]]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem_CT_mask(dims[3]* 2, 1,dims[3],kernel_size=3, spilt_list=spilt_list[2], patch_size=(2,2), mlp_dim=int(dims[3] * 2),mask=[mask[0][0],mask[2][1]])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem_CT_mask(dims[2], 1,dims[2],kernel_size=3,spilt_list=spilt_list[3], patch_size=(2,2), mlp_dim=int(dims[2] * 2),mask=[mask[0][0],mask[3][1]]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvit4_CIT(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",spilt_list=[[48,48,32,32//2*32//2],[48,48,32,16//2*16//2],[48,48,32,32//2*32//2],[24,24,16,64//2*64//2]],input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem_CIT(dims[2], 1,dims[3],kernel_size=3,spilt_list=spilt_list[0], patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem_CIT(dims[3], 1,dims[4],kernel_size=3,spilt_list=spilt_list[1],  patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem_CIT(dims[3]* 2, 1,dims[3],kernel_size=3, spilt_list=spilt_list[2], patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem_CIT(dims[2], 1,dims[2],kernel_size=3,spilt_list=spilt_list[3], patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class MKtinyvit3_1(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktinytem2(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktinytem2(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktinytem2(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktinytem2(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



MobileViT_dict={
    "MobileViTBlocktem1SCPE":MobileViTBlocktem1SCPE,
    "MobileViTBlocktem2SCPE":MobileViTBlocktem2SCPE,
    "MobileViTBlocktem3SCPE":MobileViTBlocktem3SCPE,
    "MobileViTBlocktem4SCPE":MobileViTBlocktem4SCPE,

}

class MKtinyvit4_tem(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",MobileViT="MobileViTBlocktem1SCPE",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]
        MobileViTBlock = MobileViT_dict[MobileViT]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlock(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),pos_len=32//2*32//2))
        self.encoder5 = nn.Sequential(MobileViTBlock(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2),pos_len=16//2*16//2))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlock(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),pos_len=32//2*32//2)
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2),pos_len=64//2*64//2))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class MKtinyvit5_tem(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",MobileViT="MobileViTBlocktem1SCPE",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]
        MobileViTBlock = MobileViT_dict[MobileViT]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlock(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),pos_len=32//2*32//2))
        self.encoder5 = nn.Sequential(MobileViTBlock(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2),pos_len=16//2*16//2))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlock(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),pos_len=32//2*32//2)
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(MobileViTBlock(dims[2]*2, 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2),pos_len=64//2*64//2))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvit6_tem(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",MobileViT="MobileViTBlocktem1SCPE",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]
        MobileViTBlock = MobileViT_dict[MobileViT]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlock(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),pos_len=32//2*32//2))
        self.encoder5 = nn.Sequential(MobileViTBlock(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2),pos_len=16//2*16//2))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlock(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),pos_len=32//2*32//2)
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv41 = MobileViTBlock(dims[2]*2, 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2),pos_len=64//2*64//2)
        self.Up_conv42 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d40 = self.Up_conv41(d4)
        d4 = torch.cat((x3, d40), dim=1)
        d4 = self.Up_conv42(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvittemtem4(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktemtem(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktemtem(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktemtem(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktemtem(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvittemnew4(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktemnew(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktemnew(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktemnew(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktemnew(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class MKtinyvittemtem4_SCPE(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktemtemSCPE(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktemtemSCPE(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktemtemSCPE(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktemtemSCPE(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class MKtinyvit2SCPE(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(MobileViTBlocktemSCPE(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2),pos_len=16//2*16//2))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktemSCPE(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),pos_len=32//2*32//2)
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1






class MKtinyvitQV2SCPE(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],attention="AttentionQV1_HH"):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        CMUNeXtBlockmodel=Block_dict[model]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(MobileViTBlockQVtemSCPE(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2),pos_len=16//2*16//2,hw=[int(16//2),int(16//2)],attention=attention))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlockQVtemSCPE(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2),pos_len=32//2*32//2,hw=[int(32//2),int(32//2)],attention=attention)
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_MK_resiual_VIT_1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_MK_resiual_VIT_1_v1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)),conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MK_resiual(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)),conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_MMK(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_MMK_v2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[32, 64, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MMK(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_MKK(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MKUP(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MKUP(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MKUP(ch_in=dims[1], ch_out=dims[1]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MKUP(ch_in=dims[2], ch_out=dims[2]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MKUP(ch_in=dims[3], ch_out=dims[3]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_MKUP(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[2], ch_out=dims[3]) for i in range(depths[3])],conv_1x1_bn(dims[2],dims[3]))
        self.encoder5 = nn.Sequential(*[CMUNeXtBlock_MK(ch_in=dims[3], ch_out=dims[4]) for i in range(depths[4])],conv_1x1_bn(dims[3],dims[4]))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = CMUNeXtBlock_MKUP(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = CMUNeXtBlock_MKUP(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = CMUNeXtBlock_MKUP(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = CMUNeXtBlock_MKUP(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_CNNVIT_v1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 128], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]),CNNTblock(in_channels=dims[1], out_channels=dims[1],kernel_sizes=[5, 9,13]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]),CNNTblock(in_channels=dims[2], out_channels=dims[2],kernel_sizes=[5, 9,13]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),CNNTblock(in_channels=dims[3], out_channels=dims[3],kernel_sizes=[5, 9,13]))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),CNNTblock(in_channels=dims[4], out_channels=dims[4],kernel_sizes=[5, 9,13]))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_CNNVIT_v2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),CNNTblock(in_channels=dims[3], out_channels=dims[3]))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),CNNTblock(in_channels=dims[4], out_channels=dims[4]))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1





class CMUNeXt_CNNVIT_v3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),CNNTblock(in_channels=dims[4], out_channels=dims[4]))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),CNNTblock(in_channels=dims[3], out_channels=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),SCSA(dims[2],8))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_CNNVIT_v4(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),CNNTblock(in_channels=dims[4], out_channels=dims[4]))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),CNNTblock(in_channels=dims[3], out_channels=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_VIT_v1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))


        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1






class CMUNeXt_VIT_v2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))


        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_VIT_v3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(MobileViTBlock(dims[3] * 2, 1,dims[3] * 2,kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2 * 2)),fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1





class CMUNeXt_VIT_v4(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_CNNVIT_v5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),CNNTblock(in_channels=dims[4], out_channels=dims[4]))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_CBAM_v0(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],cbam_reisual=False):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        self.cbam5 = CBAM(dims[4])
        self.cbam_reisual=cbam_reisual
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)
        if self.cbam_reisual:
            x5=self.cbam5(x5)+x5
        else:
            x5=self.cbam5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_CBAM_v1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),CBAM(dims[4]))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),CBAM(dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class CMUNeXt_CBAM_v4(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),CBAM(dims[3]))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),CBAM(dims[4]))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),CBAM(dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_for_skip(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_for_skip = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[1], k=kernels[1])


        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((self.encoder2_for_skip(x1), d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_for_skip_3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder2_for_skip = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder3_for_skip = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[1], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder4_for_skip = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[2], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((self.encoder4_for_skip(x3), d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((self.encoder3_for_skip(x2), d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((self.encoder2_for_skip(x1), d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_SCSA_v6(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),SCSA(dims[4],8))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class CMUNeXt_SCSA_v7(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),SCSA(dims[3],8))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_SCSA1_v1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),SCSA1(dims[4],8))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),SCSA1(dims[3],8))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_SCSA_v1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),SCSA(dims[4],8))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),SCSA(dims[3],8))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_SCSA_v1_1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),SCSA(dims[4],8))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(SCSA(dims[3]* 2,8),fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class CMUNeXt_SCSA_v1_2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(SCSA(dims[3]* 2,8),fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_SCSA_v2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0]),SCSA(dims[0],8))
        self.encoder2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1]),SCSA(dims[1],8))
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]),SCSA(dims[2],8))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),SCSA(dims[3],8))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),SCSA(dims[4],8))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),SCSA(dims[3],8))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),SCSA(dims[2],8))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = nn.Sequential(fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1]),SCSA(dims[1],8))
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = nn.Sequential(fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0]),SCSA(dims[0],8))
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_SCSA_v3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),SCSA(dims[4],8),SCSA(dims[4],8))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),SCSA(dims[3],8),SCSA(dims[3],8))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_SCSA_v4(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),SCSA(dims[3],8))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),SCSA(dims[4],8))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),SCSA(dims[3],8))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_SCSA_v5(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = nn.Sequential(CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2]),SCSA(dims[2],8))
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),SCSA(dims[3],8))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),SCSA(dims[4],8))

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),SCSA(dims[3],8))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),SCSA(dims[2],8))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)


        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_ver3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256,256], depths=[1, 1, 1, 3, 1,1], kernels=[3, 3, 7, 7, 7,7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        self.encoder6 = CMUNeXtBlock(ch_in=dims[4], ch_out=dims[5], depth=depths[5], k=kernels[5])

        # Decoder
        self.Up6 = up_conv(ch_in=dims[5], ch_out=dims[4])
        self.Up_conv6 = fusion_conv(ch_in=dims[4] * 2, ch_out=dims[4])
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)
        x6 = self.Maxpool(x5)
        x6 = self.encoder6(x6)

        d6 = self.Up6(x6)
        d6 = torch.cat((x5, d6), dim=1)
        d6 = self.Up_conv6(d6)


        d5 = self.Up5(d6)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1




class CMUNeXt_ver1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 7, 3, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMUNeXt_ver1, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class CMUNeXt_ver2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 2, 2, 2], kernels=[3, 7, 3, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMUNeXt_ver2, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



def cmunext(dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels)

def cmunext_s(dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels)


def cmunext_l(dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels)