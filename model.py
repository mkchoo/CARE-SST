import torch
import numpy as np
from functools import wraps
from packaging import version
from collections import namedtuple
from functools import partial
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# constants

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out

def exists(x):
    return x is not None

class GlobalResponseNorm(nn.Module):  # from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x




class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = LayerNorm2d(in_channels, elementwise_affine=False, eps=1e-6)     
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class ResBlock(nn.Module):
    def __init__(self, c, c_emb = None, c_skip=0, kernel_size=7, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(c + c_skip, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        self.norm = LayerNorm2d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c)
        )
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(c_emb, c)
        ) if exists(c_emb) else None

    def forward(self, x, t=None, x_skip=None):
        x_res = x
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.depthwise(x)
        if t is not None:
            emb = self.mlp(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            x = x + emb
        x = self.norm(x).permute(0, 2, 3, 1)
        x = self.channelwise(x).permute(0, 3, 1, 2)
        return x + x_res
    

    
class Double_Convnext(nn.Module):
    
    def __init__(self, c, c_emb=256):
        super().__init__()

        self.block1 = ResBlock(c, c_emb = None, c_skip=0)
        self.block2 = ResBlock(c, c_emb = c_emb, c_skip=0)
                                       

    def forward(self, x, t):
        x = self.block1(x,None,None)
        x = self.block2(x,t,None)

        return x     


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.down = nn.Sequential(
                    LayerNorm2d(in_channels, elementwise_affine=False, eps=1e-6),
                    nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                )
        self.block1 = Double_Convnext(out_channels,emb_dim)

    def forward(self, x, t):
        x = self.down(x)
        x = self.block1(x,t)

        return x 


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, c_skip, emb_dim=256 ):
        super().__init__()
        
        self.up =  nn.Sequential(
                    LayerNorm2d(in_channels, elementwise_affine=False, eps=1e-6),
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    )
        self.block1 = ResBlock(out_channels, c_emb = None, c_skip=c_skip)
        #self.block2 = Double_Convnext(out_channels,emb_dim)
        self.block2 =  ResBlock(out_channels, c_emb = emb_dim, c_skip=0)

    def forward(self, x, t, skip_x):
        x = self.up(x)
        diffY = skip_x.size()[2] - x.size()[2]
        diffX = skip_x.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])        
        x = self.block1(x, None, skip_x)
        x = self.block2(x, t, None)

        return x 

    
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm2d(dim, elementwise_affine=False, eps=1e-6)  

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm2d(dim, elementwise_affine=False, eps=1e-6)  
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x_ln = self.norm(x)

        qkv = self.to_qkv(x_ln).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out) + x

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = LayerNorm2d(dim, elementwise_affine=False, eps=1e-6)  
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x_ln = self.norm(x)

        qkv = self.to_qkv(x_ln).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out) + x
    
class CARE_SST(nn.Module):
    def __init__(self, c_in=2, c_out=1, embed_dim=256, dim = 64, device="cuda"):
        super().__init__()
        
            
        self.device = device
        self.embed_dim = embed_dim

        self.inc = nn.Sequential(nn.Conv2d(c_in, dim, kernel_size=3, padding=1, bias=False),
                                LayerNorm2d(dim, elementwise_affine=False, eps=1e-6))    
        self.inc_block1 = Double_Convnext(dim,embed_dim)
        self.sa0 = LinearAttention(dim)
       
        self.down1 = Down(dim, dim*2, embed_dim)
        self.sa1 = LinearAttention(dim*2)
        
        self.down2 = Down(dim*2, dim*4, embed_dim)
        self.sa2 = LinearAttention(dim*4)
        
        self.down3 = nn.Sequential(
                    LayerNorm2d(dim*4, elementwise_affine=False, eps=1e-6),
                    nn.Conv2d(dim*4, dim*8, kernel_size=2, stride=2),)

        self.bot1_1 = ResBlock(dim*8)
        self.bot1_2 = ResBlock(dim*8)  
        self.bot_sa1 = Attention(dim*8)        

        self.bot2_1 = ResBlock(dim*8)
        self.bot2_2 = ResBlock(dim*8)   
        self.bot_sa2 = Attention(dim*8)

        self.bot3_1 = ResBlock(dim*8)
        self.bot3_2 = ResBlock(dim*8)  
        self.bot_sa3 = Attention(dim*8)

        self.up1 = Up(dim*8, dim*4, dim*4, embed_dim)
        self.sa4 = LinearAttention(dim*4)
        
        self.up2 = Up(dim*4, dim*2, dim*2, embed_dim)
        self.sa5 = LinearAttention(dim*2)

        self.up3 = Up(dim*2, dim, dim, embed_dim)
        self.sa6 = LinearAttention(dim)
        
        self.outc = nn.Conv2d(dim, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
        
    def forward(self, x,  t):
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.embed_dim)

        
        x1 = self.inc(x)
        x1 = self.inc_block1(x1,t)
        x1 = self.sa0(x1)
        
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)    
        
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)        

        x4 = self.down3(x3)
        x4 = self.bot1_1(x4)
        x4 = self.bot1_2(x4)
        x4 = self.bot_sa1(x4)

        x4 = self.bot2_1(x4)
        x4 = self.bot2_2(x4)
        x4 = self.bot_sa2(x4)
        
        x4 = self.bot3_1(x4)
        x4 = self.bot3_2(x4)
        x4 = self.bot_sa3(x4)
        
        x = self.up1(x4, t, x3)   
        x = self.sa4(x)
        
        x = self.up2(x, t, x2)
        x = self.sa5(x)

        x = self.up3(x, t,  x1)
        x = self.sa6(x)
        
        output = self.outc(x)                
        return output


if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = CARE_SST( ).to('cuda')
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(2, 2, 300, 300).to('cuda')
    t = x.new_tensor([500] * x.shape[0]).long().to('cuda')
    print(net(x, t).shape)



            