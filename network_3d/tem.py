import torch
import torch.nn as nn
# 改进一点一点来，先做基础，后面再想加进去注意力，HH*HW*WW=HW
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
    def __init__(self, ch_in, ch_out, depth=1,
                group_kernel_sizes = [3, 5, 7, 9],
    
                ): # ch_in 需要是4的倍数
        super().__init__()
        
        assert self.ch_in // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = self.ch_in // 4
        self.group_kernel_sizes = group_kernel_sizes
        self.norm_act = nn.Sequential(nn.GroupNorm(4, ch_in),nn.GELU())

        self.local_dwc = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[0], 
                padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[2], 
                padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv2d(group_chans, group_chans, kernel_size=group_kernel_sizes[3], 
                padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.conv_1x1_bn=conv_1x1_bn(ch_in,self.group_chans)
        self.conv_nxn_bn=conv_nxn_bn(self.group_chans,ch_in)


    def forward(self, x):
        b, c, h_, w_ = x.size()
        l_x, g_x_s, g_x_m, g_x_l = torch.split(x, self.group_chans, dim=1)
        x_attn = self.sa_gate(self.norm_act(torch.cat((
            self.local_dwc(l_x),
            self.global_dwc_s(g_x_s),
            self.global_dwc_m(g_x_m),
            self.global_dwc_l(g_x_l),
        ), dim=1)))
        # 然后用1*1交互一下channel，这块也可以用注意力什么的想一想
        x_attn = self.conv_1x1_bn(x_attn)
        x=self.conv_nxn_bn(x_attn)
        return x
