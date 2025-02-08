import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Reduce
from math import sqrt
import math
### 想抱一个局部归一化方法，以卷积核尺寸为局部区域，然后求取卷积核区域均值和标准差，然后采用标准归一化


# cnn+vit+cnn
# 并行卷积处理  或者 串行卷积处理
# 并行
class MultiKernelTrans(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9], stride=1):
        super().__init__()
        self.convs = [nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=k, groups=in_channels, padding=(k - 1) // 2) for k in kernel_sizes]

    def forward(self,x):
        B,C,H,W=x.shape
        results = []
        for conv_k in self.convs:
            conv = conv_k(x)
            print(conv.shape)
            conv = F.interpolate(conv, scale_factor=K, mode='nearest')
            results.append(conv)
        results.append(conv)



def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# # 旋转位置编码计算
# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor):
    
#     # 链接: https://hengsblog.top/2023/03/10/%E5%87%A0%E7%A7%8D%E5%B8%B8%E7%94%A8%E7%9A%84%20Position%20Embedding%20%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E5%8F%8Apytorch%E5%AE%9E%E7%8E%B0/

#     # xq.shape = [batch_size, seq_len, dim]
#     print(xq.shape)
#     # xq_.shape = [batch_size, seq_len, dim // 2, 2]
#     xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
#     xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
#     # 转为复数域
#     xq_ = torch.view_as_complex(xq_)
#     xk_ = torch.view_as_complex(xk_)
    
#     # 应用旋转操作，然后将结果转回实数域
#     # xq_out.shape = [batch_size, seq_len, dim]
#     print(freqs_cis.shape,xq_.shape,xk_.shape)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
#     return xq_out.type_as(xq), xk_out.type_as(xk)
def apply_rope(self, Q, K, seq_len):
        """
        Apply Rotary Position Embedding to the queries and keys.
        This rotates the query and key vectors in each head according to their positions.
        """
        # Create position encoding for RoPE
        position = torch.arange(0, seq_len, device=Q.device).float()
        dim = torch.arange(0, self.head_dim, device=Q.device).float()

        # Sine and cosine functions for RoPE encoding
        sin_pos = torch.sin(position.unsqueeze(1) / (10000 ** (dim / self.head_dim)))  # (seq_len, head_dim)
        cos_pos = torch.cos(position.unsqueeze(1) / (10000 ** (dim / self.head_dim)))  # (seq_len, head_dim)
        
        # Expanding for Q and K
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

        # Rotate Q and K
        Q_rot = Q * cos_pos + torch.roll(Q, shifts=1, dims=-1) * sin_pos
        K_rot = K * cos_pos + torch.roll(K, shifts=1, dims=-1) * sin_pos

        return Q_rot, K_rot



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
        self.dim_head = dim_head

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b,n,d=x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        if self.rope:
            # q, k = apply_rotary_emb(q, k, precompute_freqs_cis(b, n * 2))
            q, k = self.apply_rope(q, k, n)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

    def apply_rope(self, Q, K, seq_len):
        """
        Apply Rotary Position Embedding to the queries and keys.
        This rotates the query and key vectors in each head according to their positions.
        """
        # Create position encoding for RoPE
        position = torch.arange(0, seq_len, device=Q.device).float()
        dim = torch.arange(0, self.dim_head, device=Q.device).float()

        # Sine and cosine functions for RoPE encoding
        sin_pos = torch.sin(position.unsqueeze(1) / (10000 ** (dim / self.dim_head)))  # (seq_len, head_dim)
        cos_pos = torch.cos(position.unsqueeze(1) / (10000 ** (dim / self.dim_head)))  # (seq_len, head_dim)
        
        # Expanding for Q and K
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

        # Rotate Q and K
        Q_rot = Q * cos_pos + torch.roll(Q, shifts=1, dims=-1) * sin_pos
        K_rot = K * cos_pos + torch.roll(K, shifts=1, dims=-1) * sin_pos

        return Q_rot, K_rot
    

class MultiHeadSelfAttention(nn.Module):
    #dim_in: int  
    # input dimension
    #dim_k: int   
    # key and query dimension
    #dim_v: int   
    # value dimension
    #num_heads: int  
    # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8,rope=False):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        self.rope=rope
        self.dim_head = dim_k % num_heads

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        # print(x.shape)
        batch, n, dim_in = x.shape
        # print(dim_in,self.dim_in)
        assert dim_in == self.dim_in
        nh = self.num_heads
        dk = self.dim_k // nh  
        # dim_k of each head
        dv = self.dim_v // nh  
        # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  
        # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  
        # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  
        # (batch, nh, n, dv)
        if self.rope:
            q, k =  q, k = self.apply_rope(q, k, n)
        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  
        # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  
        # batch, nh, n, n

        att = torch.matmul(dist, v)  
        # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  
        # batch, n, dim_v
        return att
    
    
    def apply_rope(self, Q, K, seq_len):
        """
        Apply Rotary Position Embedding to the queries and keys.
        This rotates the query and key vectors in each head according to their positions.
        """
        # Create position encoding for RoPE
        position = torch.arange(0, seq_len, device=Q.device).float()
        dim = torch.arange(0, self.dim_head, device=Q.device).float()

        # Sine and cosine functions for RoPE encoding
        sin_pos = torch.sin(position.unsqueeze(1) / (10000 ** (dim / self.dim_head)))  # (seq_len, head_dim)
        cos_pos = torch.cos(position.unsqueeze(1) / (10000 ** (dim / self.dim_head)))  # (seq_len, head_dim)
        
        # Expanding for Q and K
        sin_pos = sin_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        cos_pos = cos_pos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

        # Rotate Q and K
        Q_rot = Q * cos_pos + torch.roll(Q, shifts=1, dims=-1) * sin_pos
        K_rot = K * cos_pos + torch.roll(K, shifts=1, dims=-1) * sin_pos

        return Q_rot, K_rot
    
    
    

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.,attention_type='Multi',pos_embedding="sin"):
        super().__init__()
        self.layers = nn.ModuleList([])
        if pos_embedding == "Rope":
            rope=True
        else:
            rope=False
           
        if attention_type == 'Multi':
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    MultiHeadSelfAttention(dim_in=dim, dim_k=heads*dim_head, dim_v=heads*dim_head, num_heads=heads,rope=rope),
                    FeedForward(dim, mlp_dim, dropout)
                ]))

        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim, heads, dim_head, dropout,rope=rope),
                    FeedForward(dim, mlp_dim, dropout)
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 初始化位置编码矩阵
        self.embed_size = embed_size
        
        # 生成位置编码矩阵
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))  # (embed_size/2)
        
        # 计算sin和cos值并填充位置编码矩阵
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列
        
        # 增加一个维度，使其可以作为向量输入
        pe = pe.unsqueeze(0)  # (1, max_len, embed_size)
        
        # 将生成的编码矩阵注册为参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 输入的张量，形状为 (batch_size, seq_len, embed_size)
        """
        seq_len = x.size(1)
        
        # 截取或扩展位置编码，确保长度与输入的序列长度一致
        return self.pe[:, :seq_len, :]


class Trans(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, pos_embedding="sin",dropout=0.,attention_type='Multi'):
        super().__init__()
        self.pos_embedding=pos_embedding
        if pos_embedding == "sin":
            self.pos_embedding_model = PositionalEncoding(embed_size=dim_head*heads, max_len=5000)
        elif pos_embedding == "Rope":
            pass
        else:
            print("model pos emdeddin error")
            exit()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,attention_type,pos_embedding)
    def forward(self, x):
        
        if self.pos_embedding == "sin":
            x =x+ self.pos_embedding_model(x)
        # print(x.shape)
        x = self.transformer(x)
        return x


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
#########################################
class CNNTblock(nn.Module):
    def __init__(self, in_channels, out_channels,CNNTparam=[1,8,"sin",0,'Multi'] ,kernel_sizes=[3, 5, 7, 9], stride=1):
        super().__init__()
        self.convs = nn.Sequential(*[nn.Conv2d(in_channels, in_channels, kernel_size=k, stride=k, groups=in_channels, padding=(k - 1) // 2) for k in kernel_sizes])
        self.stride = stride
        self.kernel_sizes=kernel_sizes
        self.trans=nn.Sequential(*[Trans(in_channels, depth=CNNTparam[0], heads=CNNTparam[1], dim_head=in_channels//CNNTparam[1], 
                mlp_dim=in_channels//CNNTparam[1], pos_embedding=CNNTparam[2],dropout=CNNTparam[3],attention_type=CNNTparam[4]) for k in kernel_sizes])
        self.conv_1=conv_1x1_bn(in_channels*len(kernel_sizes), in_channels)
        self.conv_2=conv_1x1_bn(in_channels*2, out_channels)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B,C,H,W=x.shape
        results = []
        for k in range(len(self.convs)):
            conv = self.convs[k](x)
            b,c,h,w=conv.shape
            # print(conv.shape,self.kernel_sizes[k])
            # results.append(conv)
            trans=self.trans[k](conv.view(b, c, -1).transpose(1, 2))
            trans = rearrange(trans, 'b (h w) c -> b c h w', h=h)
            trans=F.interpolate(trans, scale_factor=self.kernel_sizes[k], mode='nearest')
            _,_,hh,ww=trans.shape
            trans=trans[:,:,(hh-H)//2:H+(hh-H)//2,(ww-W)//2:W+(ww-W)//2]
            # print(trans.shape,self.kernel_sizes[k])
            results.append(trans)
            # 这块的融合可以先参考mobilevit，还有一种就是可以考虑用trans得到的N*N的注意力矩阵，就相当于得到了每个patch的相关性，然后再用原feature去作为查询（一个patch内的作为一个），不过要考虑维度问题
        out=torch.cat(results, dim=1)
        out=self.conv_1(out)
        out=torch.cat([out,x], dim=1)
        out=self.conv_2(out)
        # Concatenate the results along the channel dimension
        return out

def conv_2d_bn(inp, oup, kernel_size=3, stride=1,groups=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, groups=groups, padding=padding),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
class CNNTblock1(nn.Module): # 加batchnorm 和激活函数
    def __init__(self, in_channels, out_channels,CNNTparam=[1,8,"sin",0,'Multi'] ,kernel_sizes=[3, 5], stride=1):
        super().__init__()
        self.convs = nn.Sequential(*[conv_2d_bn(in_channels, in_channels, kernel_size=k, stride=k, groups=in_channels, padding=(k - 1) // 2) for k in kernel_sizes])
        self.stride = stride
        self.kernel_sizes=kernel_sizes
        self.trans=nn.Sequential(*[Trans(in_channels, depth=CNNTparam[0], heads=CNNTparam[1], dim_head=in_channels//CNNTparam[1], 
                mlp_dim=in_channels//CNNTparam[1], pos_embedding=CNNTparam[2],dropout=CNNTparam[3],attention_type=CNNTparam[4]),nn.BatchNorm2d(in_channels),nn.SiLU()],
                *[Trans(in_channels, depth=CNNTparam[0], heads=CNNTparam[1], dim_head=in_channels//CNNTparam[1], 
                mlp_dim=in_channels//CNNTparam[1], pos_embedding=CNNTparam[2],dropout=CNNTparam[3],attention_type=CNNTparam[4]),nn.BatchNorm2d(in_channels),nn.SiLU()],
                *[Trans(in_channels, depth=CNNTparam[0], heads=CNNTparam[1], dim_head=in_channels//CNNTparam[1], 
                mlp_dim=in_channels//CNNTparam[1], pos_embedding=CNNTparam[2],dropout=CNNTparam[3],attention_type=CNNTparam[4]),nn.BatchNorm2d(in_channels),nn.SiLU()])
        self.conv_1=conv_1x1_bn(in_channels*len(kernel_sizes), in_channels)
        self.conv_2=conv_1x1_bn(in_channels*2, out_channels)

    def forward(self, x):
        # x shape: (B, C, H, W)
        B,C,H,W=x.shape
        results = []
        for k in range(len(self.convs)):
            conv = self.convs[k](x)
            b,c,h,w=conv.shape
            # print(conv.shape,self.kernel_sizes[k])
            # results.append(conv)
            # print(k,x.shape,conv.shape)
            trans=self.trans[k](conv.view(b, c, -1).transpose(1, 2))
            trans = rearrange(trans, 'b (h w) c -> b c h w', h=h)
            trans=F.interpolate(trans, scale_factor=self.kernel_sizes[k], mode='nearest')
            _,_,hh,ww=trans.shape
            trans=trans[:,:,(hh-H)//2:H+(hh-H)//2,(ww-W)//2:W+(ww-W)//2]
            # print(trans.shape,self.kernel_sizes[k])
            results.append(trans)
            # 这块的融合可以先参考mobilevit，还有一种就是可以考虑用trans得到的N*N的注意力矩阵，就相当于得到了每个patch的相关性，然后再用原feature去作为查询（一个patch内的作为一个），不过要考虑维度问题
        out=torch.cat(results, dim=1)
        out=self.conv_1(out)
        out=torch.cat([out,x], dim=1)
        out=self.conv_2(out)
        # Concatenate the results along the channel dimension
        return out



# Example usage:
B, C, H, W = 2, 64, 256, 256  # Example input shape
in_channels = C
out_channels = 64  # Example number of output channels
# Create a random input tensor
input_tensor = torch.randn(B, C, H, W)

# # Forward pass
# model=CNNTblock(in_channels=C, out_channels=C,CNNTparam=[2,8,"sin",0,'Multi'] ,kernel_sizes=[3, 5, 7, 9], stride=1)
# output_tensor = model(input_tensor)

# print(f"Input shape: {input_tensor.shape}")
# print(f"Output shape: {output_tensor.shape}")


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(count_parameters(model))


# import torch
# import torch.nn.functional as F

# def upsample_tensor(A, K):
#     # 获取输入张量的形状
#     N, C, H, W = A.shape
    
#     # 使用interpolate进行上采样，填充模式为nearest
#     B = F.interpolate(A, scale_factor=K, mode='nearest')
    
#     return B

# # 测试
# A = torch.randn(1, 1, 3, 3)  # 一个示例输入张量，形状为 (1, 3, 4, 4)
# K = 3  # 上采样倍数
# print(A)
# B = upsample_tensor(A, K)
# print(B.shape)  # 输出 B 的形状，应该是 (1, 3, 8, 8)
# # print(B)

