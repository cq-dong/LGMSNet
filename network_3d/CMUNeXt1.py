import torch
import torch.nn as nn
from network.Mobilevit import MobileViTBlock


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


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
        super(CMUNeXt, self).__init__()
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

class Connect_block(nn.Module):
    def __init__(self,ch_input, ch_output):
        super().__init__()
        self.conv=fusion_conv(ch_in=ch_input, ch_out=ch_output)
    def forward(self, x,y):
        xy = torch.cat((x, y), dim=1)
        out=self.conv(xy)
        return out


class CMWNeXt(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMWNeXt, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 160], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1





class CMWNeXt_v1_unconcat(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        # self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        # self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        # self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        # self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        # self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        # self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        # x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(d2_1)
        x2_2 =  self.Maxpool(x1_2)
        # x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        # x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        # x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        # x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        # x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1

class MHSA(nn.Module):
    def __init__(self, num_heads, dim,bias=False):
        super().__init__()
        # Q, K, V 转换矩阵，这里假设输入和输出的特征维度相同
        self.q = nn.Linear(dim, dim,bias=bias)
        self.k = nn.Linear(dim, dim,bias=bias)
        self.v = nn.Linear(dim, dim,bias=bias)
        self.num_heads = num_heads
	
    def forward(self, x):
        B, N, C = x.shape
        # 生成转换矩阵并分多头
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        
        # 点积得到attention score
        attn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)
        attn = attn.softmax(dim=-1)
        
        # 乘上attention score并输出
        v = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return v

class PreNorm(nn.Module):
    '''
    :param  dim 输入维度
            fn 前馈网络层，选择Multi-Head Attn和MLP二者之一
    '''
    def __init__(self, dim, fn):
        super().__init__()
        # LayerNorm: ( a - mean(last 2 dim) ) / sqrt( var(last 2 dim) )
        # 数据归一化的输入维度设定，以及保存前馈层
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    # 前向传播就是将数据归一化后传递给前馈层
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)       
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        # 设定depth个encoder相连，并添加残差结构
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # self.layers.append(nn.ModuleList([PreNorm(dim, MHSA(num_heads = heads, dim=dim))]))
            self.layers.append(PreNorm(dim, MHSA(num_heads = heads, dim=dim)))
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
    def forward(self, x):
        # 每次取出包含Norm-attention和Norm-mlp这两个的ModuleList，实现残差结构
        for attn in self.layers:
            x = attn.forward(x) + x
        return x

class RCSA(nn.Module):
    def __init__(self, dim=64,device='cuda:0'):
        super().__init__()
        # self.out_fushion = FusionLayer(3*dim, 3*dim)  
        # self.out_conv2 = nn.Conv2d(2*dim, dim, 3, 1, 1)
        # self.prelu   = torch.nn.PReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.transformer_max=Transformer(dim=2*dim,depth=1,heads=8)
        self.transformer_avg=Transformer(dim=2*dim,depth=1,heads=8)
        

        self.para = torch.nn.Parameter(torch.ones((1,2*dim,1,1)).to(device).requires_grad_()/2)
    def attention(self,feature):
        #feature  [b,c,h,w]
        b,c,h,w=feature.shape
        H_avg = feature.sum(dim=2,keepdim=False)/h #[b,c,w]
        W_avg = feature.sum(dim=3,keepdim=False)/w #[b,c,h]
        H_max,_ = feature.max(dim=2,keepdim=False) #[b,c,w]
        W_max,_ = feature.max(dim=3,keepdim=False) #[b,c,h]
        ###transformer的输入格式  [batch ,xiangliang_num, xiangliang_dim]->[b,h/w,c]
        H_avg_atten=self.transformer_avg(H_avg.permute(0,2,1)).permute(0,2,1) #[b,c,w]-[b,w,c]-[b,c,w]
        W_avg_atten=self.transformer_avg(W_avg.permute(0,2,1)).permute(0,2,1) #[b,c,h]-[b,h,c]-[b,c,h]
        H_max_atten=self.transformer_max(H_max.permute(0,2,1)).permute(0,2,1) #[b,c,w]-[b,w,c]-[b,c,w]
        W_max_atten=self.transformer_max(W_max.permute(0,2,1)).permute(0,2,1) #[b,c,h]-[b,h,c]-[b,c,h]

        H_max_atten=H_max_atten.unsqueeze(dim=2) #[b,c,1,w]
        W_max_atten=W_max_atten.unsqueeze(dim=3) #[b,c,h,1]
        H_avg_atten=H_avg_atten.unsqueeze(dim=2) #[b,c,1,w]
        W_avg_atten=W_avg_atten.unsqueeze(dim=3) #[b,c,h,1]
        hw_max=(W_max_atten@H_max_atten) #[b,c,h,1]@[b,c,1,w]->[b,c,h,w]
        hw_avg=(W_avg_atten@H_avg_atten) #[b,c,h,1]@[b,c,1,w]->[b,c,h,w]
        ###可以考虑变成hwmax*W+hw_acg*(1-w),把w设置成为参数
        ###为保证para始终在（0，1）之间，是不是应该加clip
        para=nn.Sigmoid()(self.para)
        out=torch.mul(hw_avg,para)+torch.mul(hw_max,(1-para))
        return out
        # return (hw_max+hw_avg)/2
    def forward(self,f_endecoder):
      
        fusion_all=self.attention(f_endecoder)+f_endecoder

        return fusion_all


class CMWNeXt_v1_rcsa(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.rcsa5_1=RCSA(dim=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.rcsa4_1=RCSA(dim=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.rcsa3_1=RCSA(dim=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.rcsa2_1=RCSA(dim=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.rcsa5_2=RCSA(dim=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.rcsa4_2=RCSA(dim=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.rcsa3_2=RCSA(dim=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.rcsa2_2=RCSA(dim=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.rcsa5_1(d5_1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.rcsa4_1(d4_1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.rcsa3_1(d3_1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.rcsa2_1(d2_1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.rcsa5_2(d5_2)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.rcsa4_2(d4_2)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.rcsa3_2(d3_2)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.rcsa2_2(d2_2)
        # d2_2 = self.connet6(d2_1,d2_2)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1


class CMWNeXt_v1_rcsa_v4(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.rcsa5_1=RCSA(dim=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.rcsa4_1=RCSA(dim=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.rcsa3_1=RCSA(dim=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.rcsa2_1=RCSA(dim=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.rcsa5_2=RCSA(dim=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.rcsa4_2=RCSA(dim=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.rcsa3_2=RCSA(dim=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.rcsa2_2=RCSA(dim=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.rcsa5_1(d5_1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.rcsa4_1(d4_1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.rcsa3_1(d3_1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.rcsa2_1(d2_1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.rcsa5_2(d5_2)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.rcsa4_2(d4_2)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.rcsa3_2(d3_2)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.rcsa2_2(d2_2)
        # d2_2 = self.connet6(d2_1,d2_2)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v1_rcsa_v3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.rcsa5_1=RCSA(dim=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.rcsa4_1=RCSA(dim=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.rcsa3_1=RCSA(dim=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.rcsa2_1=RCSA(dim=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.rcsa5_2=RCSA(dim=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.rcsa4_2=RCSA(dim=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.rcsa3_2=RCSA(dim=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.rcsa2_2=RCSA(dim=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.rcsa5_1(d5_1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.rcsa4_1(d4_1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        # d3_1 = self.rcsa3_1(d3_1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        # d2_1 = self.rcsa2_1(d2_1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.rcsa5_2(d5_2)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.rcsa4_2(d4_2)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        # d3_2 = self.rcsa3_2(d3_2)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        # d2_2 = self.rcsa2_2(d2_2)
        # d2_2 = self.connet6(d2_1,d2_2)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1


class CMWNeXt_v1_rcsa_v2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.rcsa5_1=RCSA(dim=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.rcsa4_1=RCSA(dim=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.rcsa3_1=RCSA(dim=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.rcsa2_1=RCSA(dim=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.rcsa5_2=RCSA(dim=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.rcsa4_2=RCSA(dim=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        # self.rcsa3_2=RCSA(dim=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        # self.rcsa2_2=RCSA(dim=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.rcsa5_1(d5_1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        # d4_1 = self.rcsa4_1(d4_1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        # d3_1 = self.rcsa3_1(d3_1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        # d2_1 = self.rcsa2_1(d2_1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.rcsa5_2(d5_2)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        # d4_2 = self.rcsa4_2(d4_2)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        # d3_2 = self.rcsa3_2(d3_2)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        # d2_2 = self.rcsa2_2(d2_2)
        # d2_2 = self.connet6(d2_1,d2_2)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v1_downconv(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        self.downconv1_1 = nn.Conv2d(dims[0], dims[0], kernel_size=2, stride=2, groups=dims[0])
        self.downconv2_1 = nn.Conv2d(dims[1], dims[1], kernel_size=2, stride=2, groups=dims[1])
        self.downconv3_1 = nn.Conv2d(dims[2], dims[2], kernel_size=2, stride=2, groups=dims[2])
        self.downconv4_1 = nn.Conv2d(dims[3], dims[3], kernel_size=2, stride=2, groups=dims[3])

        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        self.downconv1_2 = nn.Conv2d(dims[0], dims[0], kernel_size=2, stride=2, groups=dims[0])
        self.downconv2_2 = nn.Conv2d(dims[1], dims[1], kernel_size=2, stride=2, groups=dims[1])
        self.downconv3_2 = nn.Conv2d(dims[2], dims[2], kernel_size=2, stride=2, groups=dims[2])
        self.downconv4_2 = nn.Conv2d(dims[3], dims[3], kernel_size=2, stride=2, groups=dims[3])

        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.downconv1_1(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.downconv2_1(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.downconv3_1(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.downconv4_1(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.downconv1_2(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.downconv2_2(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.downconv3_2(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.downconv4_2(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v1_downconv_v2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        self.downconv1_1 = nn.Conv2d(dims[0], dims[0], kernel_size=2, stride=2, groups=dims[0])
        self.downconv2_1 = nn.Conv2d(dims[1], dims[1], kernel_size=2, stride=2, groups=dims[1])
        self.downconv3_1 = nn.Conv2d(dims[2], dims[2], kernel_size=2, stride=2, groups=dims[2])
        self.downconv4_1 = nn.Conv2d(dims[3], dims[3], kernel_size=2, stride=2, groups=dims[3])

        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        self.downconv1_2 = nn.Conv2d(dims[0], dims[0], kernel_size=2, stride=2, groups=dims[0])
        self.downconv2_2 = nn.Conv2d(dims[1], dims[1], kernel_size=2, stride=2, groups=dims[1])
        self.downconv3_2 = nn.Conv2d(dims[2], dims[2], kernel_size=2, stride=2, groups=dims[2])
        self.downconv4_2 = nn.Conv2d(dims[3], dims[3], kernel_size=2, stride=2, groups=dims[3])

        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.downconv3_1(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.downconv4_1(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.downconv3_2(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.downconv4_2(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1





class CMWNeXt_v1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v1new(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])


        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])


        self.connet1_2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2_2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3_2=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4_2=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5_2=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        # self.connet6_2=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_11 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_11), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_11 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_11), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_11 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_11), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_11 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_11), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = self.connet5_2(d5_2,d5_11)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)

        d4_2 = self.Up4_2(d5_2)
        d4_2 = self.connet4_2(d4_2,d4_11)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = self.connet3_2(d3_2,d3_11)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = self.connet2_2(d2_2,d2_11)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d2_2 = self.connet1_2(d2_2,d2_1)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v1_3(nn.Module):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v1_4(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v1_3_2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 96, 128], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1




class CMWNeXt_v1_3_1_VIT_1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 64, 128], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5_1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5_2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMWNeXt_v1_3_1_VIT_2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 64, 128], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]))
        self.encoder5_1 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]))
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.encoder5_2 = nn.Sequential(CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4]),MobileViTBlock(dims[4], 1,dims[4],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[4] * 2)))
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = nn.Sequential(fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3]),MobileViTBlock(dims[3], 1,dims[3],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[3] * 2)))
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = nn.Sequential(fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2]),MobileViTBlock(dims[2], 1,dims[2],kernel_size=3, patch_size=(2,2), mlp_dim=int(dims[2] * 2)))
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1




class CMWNeXt_v1_3_1(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 64, 128], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1

# 基于标签引导，就是第一个V的output作为输入放进第二个V中
class CMWNeXt_v1_withlabel(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)

        d1_1 = self.Conv_1x1_1(d2_1)


        x1_2 = self.connet1(x1_1*d1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)
        x5_2 = self.connet5(x5_11,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1_2 = self.Conv_1x1_2(d2_2)
        # print(d2_2.shape,d1_2.shape,d1_1.shape,"d2_2.shape,d1_2.shape,d1_1.shape")
        # d1_1 = self.Conv_1x1(d2_1)
        return d1_1,d1_2



# 基于标签引导，就是第一个V的output作为输入放进第二个V中
class CMWNeXt_v2_withlabel(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 64, 128, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
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
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        x5_11 =  self.Maxpool(x4_1)
        x5_1 = self.encoder5_1(x5_11)  # 160-256

        d5_1 = self.Up5_1(x5_1)
        d5_1 = torch.cat((x4_1, d5_1), dim=1)
        d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(d5_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)

        d1_1 = self.Conv_1x1_1(d2_1)


        x1_2 = self.connet1(x1_1*d1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)

        label2=self.Maxpool(d1_1)
        x2_2 = self.connet2(x2_11*label2,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)

        label3=self.Maxpool(label2)
        x3_2 = self.connet3(x3_11*label3,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)

        label4=self.Maxpool(label3)
        x4_2 = self.connet4(x4_11*label4,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        x5_2 =  self.Maxpool(x4_2)

        label5=self.Maxpool(label4)
        x5_2 = self.connet5(x5_11*label5,x5_2)
        x5_2 = self.encoder5_2(x5_2)

        x5_2 = self.connet6(x5_1,x5_2)
       

        d5_2 = self.Up5_2(x5_2)
        d5_2 = torch.cat((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(d5_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1_2 = self.Conv_1x1_2(d2_2)
        # print(d2_2.shape,d1_2.shape,d1_1.shape,"d2_2.shape,d1_2.shape,d1_1.shape")
        # d1_1 = self.Conv_1x1(d2_1)
        return d1_1,d1_2




class CMWNeXt_down3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 64, 128, 256, 256], depths=[1, 1, 3, 1, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMWNeXt_down3, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_1 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_1 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder5_1 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        # self.Up5_1 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5_1 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_1 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_1 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_1 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_1 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_1 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_1 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])
        self.Conv_1x1_1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)


        self.encoder1_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2_2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3_2 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4_2 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder5_2 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        # self.Up5_2 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5_2 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4_2 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4_2 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3_2 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3_2 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2_2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2_2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])


        self.connet1=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet2=Connect_block(ch_input=dims[0]*2, ch_output=dims[0])
        self.connet3=Connect_block(ch_input=dims[1]*2, ch_output=dims[1])
        self.connet4=Connect_block(ch_input=dims[2]*2, ch_output=dims[2])
        self.connet5=Connect_block(ch_input=dims[3]*2, ch_output=dims[3])
        # self.connet6=Connect_block(ch_input=dims[4]*2, ch_output=dims[4])

        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        x1_1 = self.stem(x)
        x1_1 = self.encoder1_1(x1_1)  # 16-16
        x2_11 =  self.Maxpool(x1_1)    # 
        x2_1 = self.encoder2_1(x2_11)  # 16-32
        x3_11 =  self.Maxpool(x2_1)
        x3_1 = self.encoder3_1(x3_11)  # 32-128
        x4_11 =  self.Maxpool(x3_1)
        x4_1 = self.encoder4_1(x4_11)  # 128-160
        # x5_11 =  self.Maxpool(x4_1)
        # x5_1 = self.encoder5_1(x5_11)  # 160-256

        # d5_1 = self.Up5_1(x5_1)
        # d5_1 = torch.cat((x4_1, d5_1), dim=1)
        # d5_1 = self.Up_conv5_1(d5_1)

        d4_1 = self.Up4_1(x4_1)
        d4_1 = torch.cat((x3_1, d4_1), dim=1)
        d4_1 = self.Up_conv4_1(d4_1)

        d3_1 = self.Up3_1(d4_1)
        d3_1 = torch.cat((x2_1, d3_1), dim=1)
        d3_1 = self.Up_conv3_1(d3_1)

        d2_1 = self.Up2_1(d3_1)
        d2_1 = torch.cat((x1_1, d2_1), dim=1)
        d2_1 = self.Up_conv2_1(d2_1)


        x1_2 = self.connet1(x1_1,d2_1)
        x1_2 = self.encoder1_2(x1_2)
        x2_2 =  self.Maxpool(x1_2)
        x2_2 = self.connet2(x2_11,x2_2)
        x2_2 = self.encoder2_2(x2_2)
        x3_2 =  self.Maxpool(x2_2)
        x3_2 = self.connet3(x3_11,x3_2)
        x3_2 = self.encoder3_2(x3_2)
        x4_2 =  self.Maxpool(x3_2)
        x4_2 = self.connet4(x4_11,x4_2)
        x4_2 = self.encoder4_2(x4_2)
        # x5_2 =  self.Maxpool(x4_2)
        # x5_2 = self.connet5(x5_11,x5_2)
        # x5_2 = self.encoder5_2(x5_2)

        # x5_2 = self.connet6(x5_1,x5_2)
       

        # d5_2 = self.Up5_2(x5_2)
        # d5_2 = torch.cat((x4_2, d5_2), dim=1)
        # d5_2 = self.Up_conv5_2(d5_2)
        d4_2 = self.Up4_2(x4_2)
        d4_2 = torch.cat((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4_2(d4_2)
        d3_2 = self.Up3_2(d4_2)
        d3_2 = torch.cat((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3_2(d3_2)
        d2_2 = self.Up2_2(d3_2)
        d2_2 = torch.cat((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2_2(d2_2)
        d1 = self.Conv_1x1(d2_2)

        # d1_1 = self.Conv_1x1(d2_1)
        return d1



class CMUNeXt_down3(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMUNeXt_down3, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
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
        # x5 = self.Maxpool(x4)
        # x5 = self.encoder5(x5)

        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
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


class CMUNeXt_down2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMUNeXt_down2, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        # self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
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
        # x4 = self.Maxpool(x3)
        # x4 = self.encoder4(x4)
        # x5 = self.Maxpool(x4)
        # x5 = self.encoder5(x5)

        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        # d4 = self.Up4(x4)
        # d4 = torch.cat((x3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1


class CMUNeXt_down2_large(nn.Module):
    def __init__(self, input_channel=3, num_classes=1, dims=[32, 64, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 7, 7, 7, 7]):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks
        """
        super(CMUNeXt_down2_large, self).__init__()
        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        # self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        # self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])
        # Decoder
        # self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        # self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        # self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        # self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
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
        # x4 = self.Maxpool(x3)
        # x4 = self.encoder4(x4)
        # x5 = self.Maxpool(x4)
        # x5 = self.encoder5(x5)

        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        # d4 = self.Up4(x4)
        # d4 = torch.cat((x3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.Up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1

def cmunext(input_channel=3, num_classes=1, dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_s(input_channel=3, num_classes=1, dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)


def cmunext_l(input_channel=3, num_classes=1, dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]):
    return CMUNeXt(dims=dims, depths=depths, kernels=kernels, input_channel=3, num_classes=1)