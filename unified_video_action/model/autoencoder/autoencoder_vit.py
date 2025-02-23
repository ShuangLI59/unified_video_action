import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from time import sleep
import pdb

from unified_video_action.model.autoencoder.vit_modules import TimeSformerEncoder, TimeSformerDecoder, RotaryEmbedding2

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ===========================================================================================


class ViTAutoencoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 ddconfig,
                 image_resolution,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.splits = ddconfig["splits"]
        self.s = ddconfig["timesteps"] // self.splits 
        
        print('-----------------------------------------------------------------------------------------------')
        print('time_downsample: ', ddconfig["time_downsample"])
        print('-----------------------------------------------------------------------------------------------')
        if ddconfig["time_downsample"]=='None':
            self.time_downsample = ddconfig["time_downsample"]
        else:
            self.time_downsample = int(ddconfig["time_downsample"])
            self.s = self.s // self.time_downsample
            
        self.res = image_resolution
        self.autoencoder_project_pos_ebm = ddconfig["autoencoder_project_pos_ebm"]
        print('-----------------------------------------------------------------------------------------------')
        print('autoencoder_project_pos_ebm: ', self.autoencoder_project_pos_ebm)
        print('-----------------------------------------------------------------------------------------------')
        self.embed_dim = embed_dim
        self.image_key = image_key
        
        patch_size = ddconfig['patch_size']
        if patch_size == 4:
            self.down = 2
        elif patch_size == 8:
            self.down = 3
        elif patch_size == 16:
            self.down = 4

        # if self.res == 128:
        #     patch_size = 4
        #     self.down = 2
        # elif self.res==256:
        #     patch_size = 8
        #     self.down = 3
        # elif self.res == 512:
        #     patch_size = 16
        #     self.down = 4
        # else:
        #     patch_size = 8
        #     self.down = 3
        #     # patch_size = 4
        #     # self.down = 2
        print('-----------------------------------------------------------------------------------------------')
        print('AE patch_size: ', patch_size)
        print('-----------------------------------------------------------------------------------------------')
        
        self.ddconfig = ddconfig
        self.encoder = TimeSformerEncoder(dim=ddconfig["channels"],
                                   image_size=image_resolution,
                                   num_frames=ddconfig["timesteps"],
                                   depth=8,
                                   patch_size=patch_size,
                                   time_downsample=self.time_downsample)

        self.decoder = TimeSformerDecoder(dim=ddconfig["channels"],
                                   image_size=image_resolution,
                                   num_frames=ddconfig["timesteps"],
                                   depth=8,
                                   patch_size=patch_size,
                                   time_downsample=self.time_downsample)

        if self.time_downsample=='None':
            self.to_pixel = nn.Sequential(
                Rearrange('b (t h w) c -> (b t) c h w', h=self.res // patch_size, w=self.res // patch_size),
                nn.ConvTranspose2d(ddconfig["channels"], 3, kernel_size=(patch_size, patch_size), stride=patch_size),
                )          
        else:
            self.to_pixel = nn.Sequential(
                Rearrange('b (t h w) c -> b c t h w', h=self.res // patch_size, w=self.res // patch_size),
                nn.ConvTranspose3d(
                    in_channels=ddconfig["channels"], 
                    out_channels=3, 
                    kernel_size=(self.time_downsample, patch_size, patch_size), 
                    stride=(self.time_downsample, patch_size, patch_size)
                ),
                Rearrange('b c t h w -> (b t) c h w')
            )

        self.act = nn.Sigmoid()
        ts = torch.linspace(-1, 1, steps=self.s).unsqueeze(-1)
        self.register_buffer('coords', ts)

        self.xy_token = nn.Parameter(torch.randn(1, 1, ddconfig["channels"]))
        self.xt_token = nn.Parameter(torch.randn(1, 1, ddconfig["channels"]))
        self.yt_token = nn.Parameter(torch.randn(1, 1, ddconfig["channels"]))

        if self.autoencoder_project_pos_ebm == 'learn':
            self.xy_pos_embedding = nn.Parameter(torch.randn(1, self.s + 1, ddconfig["channels"]))
            self.xt_pos_embedding = nn.Parameter(torch.randn(1, self.res//(2**self.down) + 1, ddconfig["channels"]))
            self.yt_pos_embedding = nn.Parameter(torch.randn(1, self.res//(2**self.down) + 1, ddconfig["channels"]))
        elif self.autoencoder_project_pos_ebm == 'fix':
            self.xy_pos_embedding = RotaryEmbedding2(ddconfig['channels'])
            self.xt_pos_embedding = RotaryEmbedding2(ddconfig['channels'])
            self.yt_pos_embedding = RotaryEmbedding2(ddconfig['channels'])

        self.xy_quant_attn = Transformer(ddconfig["channels"], 4, 4, ddconfig["channels"] // 8, 512)
        self.yt_quant_attn = Transformer(ddconfig["channels"], 4, 4, ddconfig["channels"] // 8, 512)
        self.xt_quant_attn = Transformer(ddconfig["channels"], 4, 4, ddconfig["channels"] // 8, 512)
                     
        self.pre_xy = torch.nn.Conv2d(ddconfig["channels"], self.embed_dim, 1)
        self.pre_xt = torch.nn.Conv2d(ddconfig["channels"], self.embed_dim, 1)
        self.pre_yt = torch.nn.Conv2d(ddconfig["channels"], self.embed_dim, 1)

        self.post_xy = torch.nn.Conv2d(self.embed_dim, ddconfig["channels"], 1)
        self.post_xt = torch.nn.Conv2d(self.embed_dim, ddconfig["channels"], 1)
        self.post_yt = torch.nn.Conv2d(self.embed_dim, ddconfig["channels"], 1)

    def encode(self, x):
        # x: b c t h w
        device = x.device
        b, C, T, H, W = x.size() # [2, 3, 16, 256, 256]
        x = rearrange(x, 'b c t h w -> b t c h w') # [2, 3, 16, 256, 256] -> [2, 16, 3, 256, 256]
        h = self.encoder(x) # [2, 16384, 384]

        h = rearrange(h, 'b (t h w) c -> b c t h w', t=self.s, h=self.res//(2**self.down)) # [2, 384, 16, 32, 32]
        
        h_xy = rearrange(h, 'b c t h w -> (b h w) t c') # [2048, 16, 384]
        n = h_xy.size(1)
        xy_token = repeat(self.xy_token, '1 1 d -> bhw 1 d', bhw = h_xy.size(0)) # [1, 1, 384]->[2048, 1, 384]
        h_xy = torch.cat([h_xy, xy_token], dim=1) # [2048, 17, 384]
        if self.autoencoder_project_pos_ebm == 'learn':
            h_xy += self.xy_pos_embedding[:, :(n+1)] # self.xy_pos_embedding, [1, 17, 384]
        elif self.autoencoder_project_pos_ebm == 'fix':
            h_xy += self.xy_pos_embedding(n+1, device = device)
        else:
            pdb.set_trace()
        h_xy = self.xy_quant_attn(h_xy)[:, 0]
        h_xy = rearrange(h_xy, '(b h w) c -> b c h w', b=b, h=self.res//(2**self.down)) # [2, 384, 32, 32]


        h_yt = rearrange(h, 'b c t h w -> (b t w) h c') # [1024, 32, 384]
        n = h_yt.size(1)
        yt_token = repeat(self.yt_token, '1 1 d -> btw 1 d', btw = h_yt.size(0))
        h_yt = torch.cat([h_yt, yt_token], dim=1)
        if self.autoencoder_project_pos_ebm == 'learn':
            h_yt += self.yt_pos_embedding[:, :(n+1)]
        elif self.autoencoder_project_pos_ebm == 'fix':
            h_yt += self.yt_pos_embedding(n+1, device = device)
        else:
            pdb.set_trace()
        h_yt = self.yt_quant_attn(h_yt)[:, 0]
        h_yt = rearrange(h_yt, '(b t w) c -> b c t w', b=b, w=self.res//(2**self.down)) # [2, 384, 16, 32]


        h_xt = rearrange(h, 'b c t h w -> (b t h) w c') # [1024, 32, 384]
        n = h_xt.size(1)
        xt_token = repeat(self.xt_token, '1 1 d -> bth 1 d', bth = h_xt.size(0))
        h_xt = torch.cat([h_xt, xt_token], dim=1)
        if self.autoencoder_project_pos_ebm == 'learn':
            h_xt += self.xt_pos_embedding[:, :(n+1)]
        elif self.autoencoder_project_pos_ebm == 'fix':
            h_xt += self.xt_pos_embedding(n+1, device = device)
        else:
            pdb.set_trace()
        h_xt = self.xt_quant_attn(h_xt)[:, 0]
        h_xt = rearrange(h_xt, '(b t h) c -> b c t h', b=b, h=self.res//(2**self.down)) # [2, 384, 16, 32]

        
        h_xy = self.pre_xy(h_xy) # [16, 384, 12, 12]->[16, 4, 12, 12]
        h_yt = self.pre_yt(h_yt) # [16, 384, 16, 12]->[16, 4, 16, 12]
        h_xt = self.pre_xt(h_xt) # [16, 384, 16, 12]->[16, 4, 16, 12]

        h_xy = torch.tanh(h_xy)
        h_yt = torch.tanh(h_yt)
        h_xt = torch.tanh(h_xt)

        h_xy = self.post_xy(h_xy)
        h_yt = self.post_yt(h_yt)
        h_xt = self.post_xt(h_xt)

        
        h_xy = h_xy.unsqueeze(-3).expand(-1,-1,self.s,-1, -1) # [2, 384, 32, 32] -> [2, 384, 16, 32, 32]
        h_yt = h_yt.unsqueeze(-2).expand(-1,-1,-1,self.res//(2**self.down), -1) # [2, 384, 16, 32] -> [2, 384, 16, 32, 32]
        h_xt = h_xt.unsqueeze(-1).expand(-1,-1,-1,-1,self.res//(2**self.down)) # [2, 384, 16, 32] -> [2, 384, 16, 32, 32]

        
        return h_xy + h_yt + h_xt #torch.cat([h_xy, h_yt, h_xt], dim=1)

    def decode(self, z):
        # [16, 384, 16, 12, 12]->[16, 2304, 384]
        b = z.size(0)
        dec = self.decoder(z) # [2, 384, 16, 32, 32] -> [2, 16384, 384]
        return 2*self.act(self.to_pixel(dec)).contiguous() -1 # [2, 16, 3, 256, 256], -1,1

    def forward(self, input):
        input = rearrange(input, 'b c (n t) h w -> (b n) c t h w', n=self.splits) # [4, 3, 16, 256, 256]
        # [16, 3, 16, 96, 96]
        
        z = self.encode(input) # [2, 16, 3, 256, 256] -> [2, 384, 16, 32, 32]
        # [16, 384, 16, 12, 12]
        dec = self.decode(z) # [2, 384, 16, 32, 32] -> [2, 16, 3, 256, 256]
        # [256, 3, 96, 96]
        return dec, 0.

    def extract(self, x):
        
        device = x.device
        b, C, T, H, W = x.size() # [1, 3, 16, 256, 256]
        x = rearrange(x, 'b c t h w -> b t c h w') # [1, 16, 3, 256, 256]
        h = self.encoder(x) # [1, 16384, 384]

        
        h = rearrange(h, 'b (t h w) c -> b c t h w', t=self.s, h=self.res//(2**self.down)) # [1, 384, 16, 32, 32], h=self.res before

        h_xy = rearrange(h, 'b c t h w -> (b h w) t c') # [1024, 16, 384]
        n = h_xy.size(1)
        xy_token = repeat(self.xy_token, '1 1 d -> bhw 1 d', bhw = h_xy.size(0))
        h_xy = torch.cat([h_xy, xy_token], dim=1)
        if self.autoencoder_project_pos_ebm == 'learn':
            h_xy += self.xy_pos_embedding[:, :(n+1)]
        elif self.autoencoder_project_pos_ebm == 'fix':
            h_xy += self.xy_pos_embedding(n+1, device = device)
        else:
            pdb.set_trace()
        h_xy = self.xy_quant_attn(h_xy)[:, 0]
        h_xy = rearrange(h_xy, '(b h w) c -> b c h w', b=b, h=self.res//(2**self.down))

        h_yt = rearrange(h, 'b c t h w -> (b t w) h c') # [512, 32, 384]
        n = h_yt.size(1)
        yt_token = repeat(self.yt_token, '1 1 d -> btw 1 d', btw = h_yt.size(0))
        h_yt = torch.cat([h_yt, yt_token], dim=1)
        if self.autoencoder_project_pos_ebm == 'learn':
            h_yt += self.yt_pos_embedding[:, :(n+1)]
        elif self.autoencoder_project_pos_ebm == 'fix':
            h_yt += self.yt_pos_embedding(n+1, device = device)
        else:
            pdb.set_trace()
        h_yt = self.yt_quant_attn(h_yt)[:, 0]
        h_yt = rearrange(h_yt, '(b t w) c -> b c t w', b=b, w=self.res//(2**self.down))

        h_xt = rearrange(h, 'b c t h w -> (b t h) w c') # [512, 32, 384]
        n = h_xt.size(1)
        xt_token = repeat(self.xt_token, '1 1 d -> bth 1 d', bth = h_xt.size(0))
        h_xt = torch.cat([h_xt, xt_token], dim=1)
        if self.autoencoder_project_pos_ebm == 'learn':
            h_xt += self.xt_pos_embedding[:, :(n+1)]
        elif self.autoencoder_project_pos_ebm == 'fix':
            h_xt += self.xt_pos_embedding(n+1, device = device)
        else:
            pdb.set_trace()
        h_xt = self.xt_quant_attn(h_xt)[:, 0]
        h_xt = rearrange(h_xt, '(b t h) c -> b c t h', b=b, h=self.res//(2**self.down))

        
        h_xy = self.pre_xy(h_xy) # [1, 384, 32, 32] -> [1, 4, 32, 32]
        h_yt = self.pre_yt(h_yt) # [1, 384, 16, 32] -> [1, 4, 16, 32]
        h_xt = self.pre_xt(h_xt) # [1, 384, 16, 32] -> [1, 4, 16, 32]

        h_xy = torch.tanh(h_xy)
        h_yt = torch.tanh(h_yt)
        h_xt = torch.tanh(h_xt)

        latent_size_x = h_xy.shape[2]
        latent_size_y = h_xy.shape[3]
        latent_size_t = h_yt.shape[2]

        h_xy = h_xy.view(h_xy.size(0), h_xy.size(1), -1)
        h_yt = h_yt.view(h_yt.size(0), h_yt.size(1), -1)
        h_xt = h_xt.view(h_xt.size(0), h_xt.size(1), -1)

        ret =  torch.cat([h_xy, h_yt, h_xt], dim=-1) # [1, 4, 1024],[1, 4, 512],[1, 4, 512] -> [1, 4, 2048]
        return ret, (latent_size_x, latent_size_y, latent_size_t)

    def decode_from_sample(self, h):
        
        latent_res = self.res // (2**self.down)
        h_xy = h[:, :, 0:latent_res*latent_res].view(h.size(0), h.size(1), latent_res, latent_res)
        h_yt = h[:, :, latent_res*latent_res:latent_res*(latent_res+self.s)].view(h.size(0), h.size(1), self.s, latent_res)
        h_xt = h[:, :, latent_res*(latent_res+self.s):].view(h.size(0), h.size(1), self.s, latent_res)

        h_xy = self.post_xy(h_xy)
        h_yt = self.post_yt(h_yt)
        h_xt = self.post_xt(h_xt)

        h_xy = h_xy.unsqueeze(-3).expand(-1,-1,self.s,-1, -1)
        h_yt = h_yt.unsqueeze(-2).expand(-1,-1,-1,self.res//(2**self.down), -1)
        h_xt = h_xt.unsqueeze(-1).expand(-1,-1,-1,-1,self.res//(2**self.down))

        z = h_xy + h_yt + h_xt

        
        b = z.size(0)
        dec = self.decoder(z)
        return 2*self.act(self.to_pixel(dec)).contiguous()-1
