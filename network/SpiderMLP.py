import torch
import torch.nn as nn
import torch.nn.functional as F
# 还能改成Transformer
# 可以统一考虑加不加残差
class MultiScaleProcessor(nn.Module):
    def __init__(self, c1, c2, c3, c4, C,mlp_hidden=8*8 + 4*4 + 2*2 + 1*1):
        super().__init__()
        self.C = C

        # 1x1 convolutions to adjust channel dimensions
        self.conv1 = nn.Conv2d(c1, C, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, C, kernel_size=1)
        self.conv3 = nn.Conv2d(c3, C, kernel_size=1)
        self.conv4 = nn.Conv2d(c4, C, kernel_size=1)

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear( (8*8 + 4*4 + 2*2 + 1*1),  mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, (8*8 + 4*4 + 2*2 + 1*1))
        )

        # 1x1 convolutions to restore original channel dimensions
        self.conv_out1 = nn.Conv2d(C, c1, kernel_size=1)
        self.conv_out2 = nn.Conv2d(C, c2, kernel_size=1)
        self.conv_out3 = nn.Conv2d(C, c3, kernel_size=1)
        self.conv_out4 = nn.Conv2d(C, c4, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        # Adjust channel dimensions
        x1 = self.conv1(x1)  # [b, C, h1, w1]
        x2 = self.conv2(x2)  # [b, C, h1/2, w1/2]
        x3 = self.conv3(x3)  # [b, C, h1/2, w1/4]
        x4 = self.conv4(x4)  # [b, C, h1/8, w1/8]

        # Reshape and flatten
        b, c, h1, w1 = x1.size()
        x1 = x1.view(b, c, -1, 8, 8).view(b, c, -1, 8*8)  # [b, C, h1/8*w1/8, 8*8]
        x2 = x2.view(b, c, -1, 4, 4).view(b, c, -1, 4*4)  # [b, C, h1/8*w1/8, 4*4]
        x3 = x3.view(b, c, -1, 2, 2).view(b, c, -1, 2*2)  # [b, C, h1/8*w1/8, 2*2]
        x4 = x4.view(b, c, -1, 1, 1).view(b, c, -1, 1*1)  # [b, C, h1/8*w1/8, 1*1]

        # Concatenate along the last dimension
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Flatten for MLP
        x = x.view(-1, 8*8 + 4*4 + 2*2 + 1*1)  # [b*C,  (8*8 + 4*4 + 2*2 + 1*1)]

        # Apply MLP
        # print(x.shape,"153")
        x = self.mlp(x)  # [b, C * (8*8 + 4*4 + 2*2 + 1*1)]

        # Reshape back to original dimensions
        x = x.view(b, c, -1, 8*8 + 4*4 + 2*2 + 1*1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Split back into individual tensors
        x1, x2, x3, x4 = x.split([8*8, 4*4, 2*2, 1*1], dim=-1)

        # Reshape back to original spatial dimensions
        x1 = x1.view(b, c, h1 // 8, w1 // 8, 8, 8).contiguous().view(b, c, h1, w1)  # [b, C, h1, w1] 
        x2 = x2.view(b, c, h1 // 8, w1 // 8, 4, 4).contiguous().view(b, c, h1 // 2, w1 // 2)  # [b, C, h1/2, w1/2]
        x3 = x3.view(b, c, h1 // 8, w1 // 8, 2, 2).contiguous().view(b, c, h1 // 4, w1 // 4)  # [b, C, h1/2, w1/4]
        x4 = x4.view(b, c, h1 // 8, w1 // 8, 1, 1).contiguous().view(b, c, h1 // 8, w1 // 8)  # [b, C, h1/8, w1/8]

        # Restore original channel dimensions
        x1 = self.conv_out1(x1)  # [b, c1, h1, w1]
        x2 = self.conv_out2(x2)  # [b, c2, h1/2, w1/2]
        x3 = self.conv_out3(x3)  # [b, c3, h1/2, w1/4]
        x4 = self.conv_out4(x4)  # [b, c4, h1/8, w1/8]

        return x1, x2, x3, x4


class MultiScaleProcessor_skip(nn.Module):
    def __init__(self, c1, c2, c3, c4, C,mlp_hidden=8*8 + 4*4 + 2*2 + 1*1):
        super().__init__()
        self.C = C

        # 1x1 convolutions to adjust channel dimensions
        self.conv1 = nn.Conv2d(c1, C, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, C, kernel_size=1)
        self.conv3 = nn.Conv2d(c3, C, kernel_size=1)
        self.conv4 = nn.Conv2d(c4, C, kernel_size=1)

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear( (8*8 + 4*4 + 2*2 + 1*1),  mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, (8*8 + 4*4 + 2*2 + 1*1))
        )

        # 1x1 convolutions to restore original channel dimensions
        self.conv_out1 = nn.Conv2d(C, c1, kernel_size=1)
        self.conv_out2 = nn.Conv2d(C, c2, kernel_size=1)
        self.conv_out3 = nn.Conv2d(C, c3, kernel_size=1)
        self.conv_out4 = nn.Conv2d(C, c4, kernel_size=1)


    def forward(self, x1_raw, x2_raw, x3_raw, x4_raw):
        # Adjust channel dimensions
        x1 = self.conv1(x1_raw)  # [b, C, h1, w1]
        x2 = self.conv2(x2_raw)  # [b, C, h1/2, w1/2]
        x3 = self.conv3(x3_raw)  # [b, C, h1/2, w1/4]
        x4 = self.conv4(x4_raw)  # [b, C, h1/8, w1/8]

        # Reshape and flatten
        b, c, h1, w1 = x1.size()
        x1 = x1.view(b, c, -1, 8, 8).view(b, c, -1, 8*8)  # [b, C, h1/8*w1/8, 8*8]
        x2 = x2.view(b, c, -1, 4, 4).view(b, c, -1, 4*4)  # [b, C, h1/8*w1/8, 4*4]
        x3 = x3.view(b, c, -1, 2, 2).view(b, c, -1, 2*2)  # [b, C, h1/8*w1/8, 2*2]
        x4 = x4.view(b, c, -1, 1, 1).view(b, c, -1, 1*1)  # [b, C, h1/8*w1/8, 1*1]

        # Concatenate along the last dimension
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Flatten for MLP
        x = x.view(-1, 8*8 + 4*4 + 2*2 + 1*1)  # [b*C,  (8*8 + 4*4 + 2*2 + 1*1)]

        # Apply MLP
        # print(x.shape,"153")
        x = self.mlp(x)  # [b, C * (8*8 + 4*4 + 2*2 + 1*1)]

        # Reshape back to original dimensions
        x = x.view(b, c, -1, 8*8 + 4*4 + 2*2 + 1*1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Split back into individual tensors
        x1, x2, x3, x4 = x.split([8*8, 4*4, 2*2, 1*1], dim=-1)

        # Reshape back to original spatial dimensions
        x1 = x1.view(b, c, h1 // 8, w1 // 8, 8, 8).contiguous().view(b, c, h1, w1)  # [b, C, h1, w1] 
        x2 = x2.view(b, c, h1 // 8, w1 // 8, 4, 4).contiguous().view(b, c, h1 // 2, w1 // 2)  # [b, C, h1/2, w1/2]
        x3 = x3.view(b, c, h1 // 8, w1 // 8, 2, 2).contiguous().view(b, c, h1 // 4, w1 // 4)  # [b, C, h1/2, w1/4]
        x4 = x4.view(b, c, h1 // 8, w1 // 8, 1, 1).contiguous().view(b, c, h1 // 8, w1 // 8)  # [b, C, h1/8, w1/8]

        # Restore original channel dimensions
        x1 = self.conv_out1(x1)+x1_raw  # [b, c1, h1, w1]
        x2 = self.conv_out2(x2)+x2_raw  # [b, c2, h1/2, w1/2]
        x3 = self.conv_out3(x3)+x3_raw  # [b, c3, h1/2, w1/4]
        x4 = self.conv_out4(x4)+x4_raw  # [b, c4, h1/8, w1/8]

        return x1, x2, x3, x4


class MultiScaleProcessor2(nn.Module):  # 与1的不同在于，这个最底层的全部feature都输进去了
    def __init__(self, c1, c2, c3, c4, C,mlp_hidden=8*8 + 4*4 + 2*2 + 16*16):
        super().__init__()
        self.C = C

        # 1x1 convolutions to adjust channel dimensions
        self.conv1 = nn.Conv2d(c1, C, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, C, kernel_size=1)
        self.conv3 = nn.Conv2d(c3, C, kernel_size=1)
        self.conv4 = nn.Conv2d(c4, C, kernel_size=1)

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear( (8*8 + 4*4 + 2*2 + 16*16),  mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, (8*8 + 4*4 + 2*2 + 16*16))
        )

        # 1x1 convolutions to restore original channel dimensions
        self.conv_out1 = nn.Conv2d(C, c1, kernel_size=1)
        self.conv_out2 = nn.Conv2d(C, c2, kernel_size=1)
        self.conv_out3 = nn.Conv2d(C, c3, kernel_size=1)
        self.conv_out4 = nn.Conv2d(C*256, c4, kernel_size=1)   # 虽然可以，但难免还是太大，没必要，弃用

    def forward(self, x1, x2, x3, x4):
        # Adjust channel dimensions
        x1 = self.conv1(x1)  # [b, C, h1, w1]
        x2 = self.conv2(x2)  # [b, C, h1/2, w1/2]
        x3 = self.conv3(x3)  # [b, C, h1/2, w1/4]
        x4 = self.conv4(x4)  # [b, C, h1/8, w1/8]

        # Reshape and flatten
        b, c, h1, w1 = x1.size()
        x1 = x1.view(b, c, -1, 8, 8).view(b, c, -1, 8*8)  # [b, C, h1/8*w1/8, 8*8]
        x2 = x2.view(b, c, -1, 4, 4).view(b, c, -1, 4*4)  # [b, C, h1/8*w1/8, 4*4]
        x3 = x3.view(b, c, -1, 2, 2).view(b, c, -1, 2*2)  # [b, C, h1/8*w1/8, 2*2]
        x4 = x4.view(b, c, -1, 16, 16).view(b, c, -1, 16*16)  # [b, C, h1/8*w1/8, 1*1]
        if x3.size(2) != x4.size(2):
            repeat_factor = x3.size(2) // x4.size(2)
        # Expand x4's third dimension to match x3's third dimension using repeat
            x4 = x4.repeat(1, 1, repeat_factor, 1).contiguous()

        # Concatenate along the last dimension
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Flatten for MLP
        x = x.view(-1, 8*8 + 4*4 + 2*2 + 16*16)  # [b*C,  (8*8 + 4*4 + 2*2 + 1*1)]

        # Apply MLP
        # print(x.shape,"153")
        x = self.mlp(x)  # [b, C * (8*8 + 4*4 + 2*2 + 1*1)]

        # Reshape back to original dimensions
        x = x.view(b, c, -1, 8*8 + 4*4 + 2*2 + 16*16)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Split back into individual tensors
        x1, x2, x3, x4 = x.split([8*8, 4*4, 2*2, 16*16], dim=-1)

        # Reshape back to original spatial dimensions
        x1 = x1.view(b, c, h1 // 8, w1 // 8, 8, 8).contiguous().view(b, c, h1, w1)  # [b, C, h1, w1] 
        x2 = x2.view(b, c, h1 // 8, w1 // 8, 4, 4).contiguous().view(b, c, h1 // 2, w1 // 2)  # [b, C, h1/2, w1/2]
        x3 = x3.view(b, c, h1 // 8, w1 // 8, 2, 2).contiguous().view(b, c, h1 // 4, w1 // 4)  # [b, C, h1/2, w1/4]
        x4 = x4.view(b, c, h1 // 8, w1 // 8, 16, 16).contiguous().view(b, c*h1 // 8*w1 // 8,16,16)  # [b, C, h1/8, w1/8]

        # Restore original channel dimensions
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        x1 = self.conv_out1(x1)  # [b, c1, h1, w1]
        x2 = self.conv_out2(x2)  # [b, c2, h1/2, w1/2]
        x3 = self.conv_out3(x3)  # [b, c3, h1/2, w1/4]
        x4 = self.conv_out4(x4)  # [b, c4, h1/8, w1/8]

        return x1, x2, x3, x4
    

class MultiScaleProcessor3(nn.Module):  # 与1的不同在于，这个最底层的全部feature都输进去了
    def __init__(self, c1, c2, c3, c4, C,mlp_hidden=8*8 + 4*4 + 2*2 + 16*16):
        super().__init__()
        self.C = C

        # 1x1 convolutions to adjust channel dimensions
        self.conv1 = nn.Conv2d(c1, C, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, C, kernel_size=1)
        self.conv3 = nn.Conv2d(c3, C, kernel_size=1)
        self.conv4 = nn.Conv2d(c4, C, kernel_size=1)

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear( (8*8 + 4*4 + 2*2 + 16*16),  mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, (8*8 + 4*4 + 2*2 + 1*1))
        )

        # 1x1 convolutions to restore original channel dimensions
        self.conv_out1 = nn.Conv2d(C, c1, kernel_size=1)
        self.conv_out2 = nn.Conv2d(C, c2, kernel_size=1)
        self.conv_out3 = nn.Conv2d(C, c3, kernel_size=1)
        self.conv_out4 = nn.Conv2d(C, c4, kernel_size=1)   # 虽然可以，但难免还是太大，没必要，弃用

    def forward(self, x1, x2, x3, x4):
        # Adjust channel dimensions
        x1 = self.conv1(x1)  # [b, C, h1, w1]
        x2 = self.conv2(x2)  # [b, C, h1/2, w1/2]
        x3 = self.conv3(x3)  # [b, C, h1/2, w1/4]
        x4 = self.conv4(x4)  # [b, C, h1/8, w1/8]

        # Reshape and flatten
        b, c, h1, w1 = x1.size()
        x1 = x1.view(b, c, -1, 8, 8).view(b, c, -1, 8*8)  # [b, C, h1/8*w1/8, 8*8]
        x2 = x2.view(b, c, -1, 4, 4).view(b, c, -1, 4*4)  # [b, C, h1/8*w1/8, 4*4]
        x3 = x3.view(b, c, -1, 2, 2).view(b, c, -1, 2*2)  # [b, C, h1/8*w1/8, 2*2]
        x4 = x4.view(b, c, -1, 16, 16).view(b, c, -1, 16*16)  # [b, C, h1/8*w1/8, 1*1]
        if x3.size(2) != x4.size(2):
            repeat_factor = x3.size(2) // x4.size(2)
        # Expand x4's third dimension to match x3's third dimension using repeat
            x4 = x4.repeat(1, 1, repeat_factor, 1).contiguous()

        # Concatenate along the last dimension
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Flatten for MLP
        x = x.view(-1, 8*8 + 4*4 + 2*2 + 16*16)  # [b*C,  (8*8 + 4*4 + 2*2 + 1*1)]

        # Apply MLP
        # print(x.shape,"153")
        x = self.mlp(x)  # [b, C * (8*8 + 4*4 + 2*2 + 1*1)]

        # Reshape back to original dimensions
        x = x.view(b, c, -1, 8*8 + 4*4 + 2*2 + 1*1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Split back into individual tensors
        x1, x2, x3, x4 = x.split([8*8, 4*4, 2*2, 1*1], dim=-1)

        # Reshape back to original spatial dimensions
        x1 = x1.view(b, c, h1 // 8, w1 // 8, 8, 8).contiguous().view(b, c, h1, w1)  # [b, C, h1, w1] 
        x2 = x2.view(b, c, h1 // 8, w1 // 8, 4, 4).contiguous().view(b, c, h1 // 2, w1 // 2)  # [b, C, h1/2, w1/2]
        x3 = x3.view(b, c, h1 // 8, w1 // 8, 2, 2).contiguous().view(b, c, h1 // 4, w1 // 4)  # [b, C, h1/2, w1/4]
        x4 = x4.view(b, c, h1 // 8, w1 // 8, 1, 1).contiguous().view(b, c,h1 // 8,w1 // 8)  # [b, C, h1/8, w1/8]

        # Restore original channel dimensions
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        x1 = self.conv_out1(x1)  # [b, c1, h1, w1]
        x2 = self.conv_out2(x2)  # [b, c2, h1/2, w1/2]
        x3 = self.conv_out3(x3)  # [b, c3, h1/2, w1/4]
        x4 = self.conv_out4(x4)  # [b, c4, h1/8, w1/8]

        return x1, x2, x3, x4
    

class MultiScaleProcessor4(nn.Module):  # 与1的不同在于，这个最底层的全部feature都输进去了
    def __init__(self, c1, c2, c3, c4, C,mlp_hidden=8*8 + 4*4 + 2*2 + 16*16):
        super().__init__()
        self.C = C

        # 1x1 convolutions to adjust channel dimensions
        self.conv1 = nn.Conv2d(c1, C, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, C, kernel_size=1)
        self.conv3 = nn.Conv2d(c3, C, kernel_size=1)
        self.conv4 = nn.Conv2d(c4, C, kernel_size=1)

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear( (8*8 + 4*4 + 2*2 + 16*16),  mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, (8*8 + 4*4 + 2*2 + 1*1))
        )

        # 1x1 convolutions to restore original channel dimensions
        self.conv_out1 = nn.Conv2d(C, c1, kernel_size=1)
        self.conv_out2 = nn.Conv2d(C, c2, kernel_size=1)
        self.conv_out3 = nn.Conv2d(C, c3, kernel_size=1)
        self.conv_out4 = nn.Conv2d(C, c4, kernel_size=1)   # 虽然可以，但难免还是太大，没必要，弃用

    def forward(self, x1, x2, x3, x4_raw):
        # Adjust channel dimensions
        x1 = self.conv1(x1)  # [b, C, h1, w1]
        x2 = self.conv2(x2)  # [b, C, h1/2, w1/2]
        x3 = self.conv3(x3)  # [b, C, h1/2, w1/4]
        x4 = self.conv4(x4_raw)  # [b, C, h1/8, w1/8]

        # Reshape and flatten
        b, c, h1, w1 = x1.size()
        x1 = x1.view(b, c, -1, 8, 8).view(b, c, -1, 8*8)  # [b, C, h1/8*w1/8, 8*8]
        x2 = x2.view(b, c, -1, 4, 4).view(b, c, -1, 4*4)  # [b, C, h1/8*w1/8, 4*4]
        x3 = x3.view(b, c, -1, 2, 2).view(b, c, -1, 2*2)  # [b, C, h1/8*w1/8, 2*2]
        x4 = x4.view(b, c, -1, 16, 16).view(b, c, -1, 16*16)  # [b, C, h1/8*w1/8, 1*1]
        if x3.size(2) != x4.size(2):
            repeat_factor = x3.size(2) // x4.size(2)
        # Expand x4's third dimension to match x3's third dimension using repeat
            x4 = x4.repeat(1, 1, repeat_factor, 1).contiguous()

        # Concatenate along the last dimension
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Flatten for MLP
        x = x.view(-1, 8*8 + 4*4 + 2*2 + 16*16)  # [b*C,  (8*8 + 4*4 + 2*2 + 1*1)]

        # Apply MLP
        # print(x.shape,"153")
        x = self.mlp(x)  # [b, C * (8*8 + 4*4 + 2*2 + 1*1)]

        # Reshape back to original dimensions
        x = x.view(b, c, -1, 8*8 + 4*4 + 2*2 + 1*1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Split back into individual tensors
        x1, x2, x3, x4 = x.split([8*8, 4*4, 2*2, 1*1], dim=-1)

        # Reshape back to original spatial dimensions
        x1 = x1.view(b, c, h1 // 8, w1 // 8, 8, 8).contiguous().view(b, c, h1, w1)  # [b, C, h1, w1] 
        x2 = x2.view(b, c, h1 // 8, w1 // 8, 4, 4).contiguous().view(b, c, h1 // 2, w1 // 2)  # [b, C, h1/2, w1/2]
        x3 = x3.view(b, c, h1 // 8, w1 // 8, 2, 2).contiguous().view(b, c, h1 // 4, w1 // 4)  # [b, C, h1/2, w1/4]
        x4 = x4.view(b, c, h1 // 8, w1 // 8, 1, 1).contiguous().view(b, c,h1 // 8,w1 // 8)  # [b, C, h1/8, w1/8]

        # Restore original channel dimensions
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
        x1 = self.conv_out1(x1)  # [b, c1, h1, w1]
        x2 = self.conv_out2(x2)  # [b, c2, h1/2, w1/2]
        x3 = self.conv_out3(x3)  # [b, c3, h1/2, w1/4]
        x4 = self.conv_out4(x4)+x4_raw  # [b, c4, h1/8, w1/8]

        return x1, x2, x3, x4
    

class MultiScaleProcessor3_skip(nn.Module):  # 与1的不同在于，这个最底层的全部feature都输进去了
    def __init__(self, c1, c2, c3, c4, C,mlp_hidden=8*8 + 4*4 + 2*2 + 16*16):
        super().__init__()
        self.C = C

        # 1x1 convolutions to adjust channel dimensions
        self.conv1 = nn.Conv2d(c1, C, kernel_size=1)
        self.conv2 = nn.Conv2d(c2, C, kernel_size=1)
        self.conv3 = nn.Conv2d(c3, C, kernel_size=1)
        self.conv4 = nn.Conv2d(c4, C, kernel_size=1)

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear( (8*8 + 4*4 + 2*2 + 16*16),  mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, (8*8 + 4*4 + 2*2 + 1*1))
        )

        # 1x1 convolutions to restore original channel dimensions
        self.conv_out1 = nn.Conv2d(C, c1, kernel_size=1)
        self.conv_out2 = nn.Conv2d(C, c2, kernel_size=1)
        self.conv_out3 = nn.Conv2d(C, c3, kernel_size=1)
        self.conv_out4 = nn.Conv2d(C, c4, kernel_size=1)   # 虽然可以，但难免还是太大，没必要，弃用


    def forward(self, x1_raw, x2_raw, x3_raw, x4_raw):
        # Adjust channel dimensions
        x1 = self.conv1(x1_raw)  # [b, C, h1, w1]
        x2 = self.conv2(x2_raw)  # [b, C, h1/2, w1/2]
        x3 = self.conv3(x3_raw)  # [b, C, h1/2, w1/4]
        x4 = self.conv4(x4_raw)  # [b, C, h1/8, w1/8]

        # Reshape and flatten
        b, c, h1, w1 = x1.size()
        x1 = x1.view(b, c, -1, 8, 8).view(b, c, -1, 8*8)  # [b, C, h1/8*w1/8, 8*8]
        x2 = x2.view(b, c, -1, 4, 4).view(b, c, -1, 4*4)  # [b, C, h1/8*w1/8, 4*4]
        x3 = x3.view(b, c, -1, 2, 2).view(b, c, -1, 2*2)  # [b, C, h1/8*w1/8, 2*2]
        x4 = x4.view(b, c, -1, 16, 16).view(b, c, -1, 16*16)  # [b, C, h1/8*w1/8, 1*1]
        if x3.size(2) != x4.size(2):
            repeat_factor = x3.size(2) // x4.size(2)
        # Expand x4's third dimension to match x3's third dimension using repeat
            x4 = x4.repeat(1, 1, repeat_factor, 1).contiguous()

        # Concatenate along the last dimension
        x = torch.cat((x1, x2, x3, x4), dim=-1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Flatten for MLP
        x = x.view(-1, 8*8 + 4*4 + 2*2 + 16*16)  # [b*C,  (8*8 + 4*4 + 2*2 + 1*1)]

        # Apply MLP
        # print(x.shape,"153")
        x = self.mlp(x)  # [b, C * (8*8 + 4*4 + 2*2 + 1*1)]

        # Reshape back to original dimensions
        x = x.view(b, c, -1, 8*8 + 4*4 + 2*2 + 1*1)  # [b, C, h1/8*w1/8, 8*8 + 4*4 + 2*2 + 1*1]

        # Split back into individual tensors
        x1, x2, x3, x4 = x.split([8*8, 4*4, 2*2, 1*1], dim=-1)

        # Reshape back to original spatial dimensions
        x1 = x1.view(b, c, h1 // 8, w1 // 8, 8, 8).contiguous().view(b, c, h1, w1)  # [b, C, h1, w1] 
        x2 = x2.view(b, c, h1 // 8, w1 // 8, 4, 4).contiguous().view(b, c, h1 // 2, w1 // 2)  # [b, C, h1/2, w1/2]
        x3 = x3.view(b, c, h1 // 8, w1 // 8, 2, 2).contiguous().view(b, c, h1 // 4, w1 // 4)  # [b, C, h1/2, w1/4]
        x4 = x4.view(b, c, h1 // 8, w1 // 8, 1, 1).contiguous().view(b, c,h1 // 8,w1 // 8)  # [b, C, h1/8, w1/8]

        # Restore original channel dimensions
        # print(x1.shape,x2.shape,x3.shape,x4.shape)
         # Restore original channel dimensions
        x1 = self.conv_out1(x1)+x1_raw  # [b, c1, h1, w1]
        x2 = self.conv_out2(x2)+x2_raw  # [b, c2, h1/2, w1/2]
        x3 = self.conv_out3(x3)+x3_raw  # [b, c3, h1/2, w1/4]
        x4 = self.conv_out4(x4)+x4_raw  # [b, c4, h1/8, w1/8]
        return x1, x2, x3, x4
    



if __name__ =="main":
# if 1:
    # Example usage:
    # c1, c2, c3, c4 are the original channel dimensions
    # C is the target channel dimension after the 1x1 convolutions
    # model = TorchProcessor(c1=64, c2=128, c3=256, c4=512, C=256)
    model = MultiScaleProcessor_skip(c1=64, c2=64, c3=64, c4=64, C=64)

    # Dummy input tensors
    x1 = torch.randn(1, 64, 128, 128)
    x2 = torch.randn(1, 64, 64, 64)
    x3 = torch.randn(1, 64, 32, 32)
    x4 = torch.randn(1, 64, 16, 16)

    # Forward pass
    x1_out, x2_out, x3_out, x4_out = model(x1, x2, x3, x4)


    print(x1_out.shape, x2_out.shape, x3_out.shape, x4_out.shape)
    # print(x1-x1_out)
    # print(x2-x2_out)
    # print(x3-x3_out)
    # print(x4-x4_out)
    # print("x1 input and output are equal:", torch.allclose(x1, x1_out, atol=1e-6))
    # print("x1 input and output are equal:", torch.allclose(x2, x2_out, atol=1e-6))
    # print("x1 input and output are equal:", torch.allclose(x3, x3_out, atol=1e-6))
    # print("x1 input and output are equal:", torch.allclose(x4, x4_out, atol=1e-6))
