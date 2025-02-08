import torch
import torch.nn as nn
from network.SCSA import SCSA,SCSA1,SCSA2,PCSA,SMSA
from network.CBAM import CBAM
from network.CNNVIT import CNNTblock,CNNTblock1
from network.Mobilevit import MobileViTBlock,MobileViTBlock1,MobileViTBlock2
from network.Mobilevit3d import MobileViTBlock3D,MobileViTBlock3DPOS,MobileViTBlocktem_CT3D

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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# 直接把 block改为
class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv3d(ch_in, ch_in, kernel_size=(k, k, k), groups=ch_in, padding=(k // 2, k // 2, k // 2)),
                    nn.GELU(),
                    nn.InstanceNorm3d(ch_in)
                )),
                nn.Conv3d(ch_in, ch_in * 4, kernel_size=(1, 1, 1)),
                nn.GELU(),
                nn.InstanceNorm3d(ch_in * 4),
                nn.Conv3d(ch_in * 4, ch_in, kernel_size=(1, 1, 1)),
                nn.GELU(),
                nn.InstanceNorm3d(ch_in)
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
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 3D 上采样使用 'trilinear' 插值
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class fusion_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.InstanceNorm3d(ch_in),
            nn.Conv3d(ch_in, ch_out * 4, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(ch_out * 4),
            nn.Conv3d(ch_out * 4, ch_out, kernel_size=(1, 1, 1)),
            nn.GELU(),
            nn.InstanceNorm3d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CMUNeXt(nn.Module):
    def __init__(self, input_channel=1, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernel size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
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
        self.Conv_1x1x1 = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

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
        d1 = self.Conv_1x1x1(d2)

        return d1

class CMUNeXt_VIT3d_1(nn.Module):
    def __init__(self, input_channel=1, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernel size of cmunext blocks
        """
        super().__init__()
        # Encoder
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),MobileViTBlock3D(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock3D(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[4] * 2)))


        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock3D(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock3D(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1x1 = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

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
        d1 = self.Conv_1x1x1(d2)

        return d1



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

        # self.local_dwc = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0], 
        #         padding=group_kernel_sizes[0] // 2, groups=self.group_chans)
        # self.global_dwc_s = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
        #         padding=group_kernel_sizes[1] // 2, groups=self.group_chans)
        # self.global_dwc_m = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2], 
        #         padding=group_kernel_sizes[2] // 2, groups=self.group_chans)
        # self.global_dwc_l = nn.Conv2d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3], 
        #         padding=group_kernel_sizes[3] // 2, groups=self.group_chans)

        # 3D卷积的局部卷积
        self.local_dwc = nn.Conv3d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0], 
                                      padding=(group_kernel_sizes[0] // 2, group_kernel_sizes[0] // 2, group_kernel_sizes[0] // 2),
                                      groups=self.group_chans)
        # 3D卷积的全局卷积，分别是小、中、大卷积
        self.global_dwc_s = nn.Conv3d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=(group_kernel_sizes[1] // 2, group_kernel_sizes[1] // 2, group_kernel_sizes[1] // 2),
                                      groups=self.group_chans)
        
        self.global_dwc_m = nn.Conv3d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[2], 
                                      padding=(group_kernel_sizes[2] // 2, group_kernel_sizes[2] // 2, group_kernel_sizes[2] // 2),
                                      groups=self.group_chans)
        
        self.global_dwc_l = nn.Conv3d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[3], 
                                      padding=(group_kernel_sizes[3] // 2, group_kernel_sizes[3] // 2, group_kernel_sizes[3] // 2),
                                      groups=self.group_chans)


        self.conv_1x1_bn=conv_1x1_bn(ch_in,self.group_chans)
        self.conv_nxn_bn=conv_nxn_bn(self.group_chans,ch_in)


    def forward(self, x):
        xclone=x.clone()
        b, c, h_, w_,f_ = x.size()
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




Block_dict={
    "CMUNeXtBlock_MK_resiual2":CMUNeXtBlock_MK_resiual2,
}


class MKtinyvit4(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual2",input_channel=1, num_classes=1, dims=[16, 32, 64, 128, 160], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlock3D(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlock3D(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlock3D(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock3D(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

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


class MKtinyvit4POS(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual2",input_channel=1, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlock3DPOS(dims[2], 1,dims[3],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlock3DPOS(dims[3], 1,dims[4],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlock3DPOS(dims[3]* 2, 1,dims[3],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock3DPOS(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

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





class MKtinyvit4_CT3D(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",spilt_list=[[96,32],[96,32],[96,32],[48,16]],input_channel=1, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem_CT3D(dims[2], 1,dims[3],kernel_size=3,spilt_list=spilt_list[0], patch_size=(2,2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem_CT3D(dims[3], 1,dims[4],kernel_size=3,spilt_list=spilt_list[1],  patch_size=(2,2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem_CT3D(dims[3]* 2, 1,dims[3],kernel_size=3, spilt_list=spilt_list[2], patch_size=(2,2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem_CT3D(dims[2], 1,dims[2],kernel_size=3,spilt_list=spilt_list[3], patch_size=(2,2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


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




class MKtinyvit4_CT3D_L(nn.Module):
    def __init__(self, model="CMUNeXtBlock_MK_resiual",spilt_list=[[96,32],[96,32],[96,32],[48,16]],input_channel=1, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[0]) for i in range(depths[0])],conv_1x1_bn(dims[0],dims[0]))
        self.encoder2 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[0], ch_out=dims[1]) for i in range(depths[1])],conv_1x1_bn(dims[0],dims[1]))
        self.encoder3 = nn.Sequential(*[CMUNeXtBlockmodel(ch_in=dims[1], ch_out=dims[2]) for i in range(depths[2])],conv_1x1_bn(dims[1],dims[2]))
        self.encoder4 = nn.Sequential(MobileViTBlocktem_CT3D(dims[2], 1,dims[3],kernel_size=3,spilt_list=spilt_list[0], patch_size=(2,2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5 = nn.Sequential(MobileViTBlocktem_CT3D(dims[3], 1,dims[4],kernel_size=3,spilt_list=spilt_list[1],  patch_size=(1,1,1), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = MobileViTBlocktem_CT3D(dims[3]* 2, 1,dims[3],kernel_size=3, spilt_list=spilt_list[2], patch_size=(2,2,2), mlp_dim=int(dims[3] * 2))
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlocktem_CT3D(dims[2], 1,dims[2],kernel_size=3,spilt_list=spilt_list[3], patch_size=(2,2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1 = nn.Conv3d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


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