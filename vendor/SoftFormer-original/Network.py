import torch
import torch.nn as nn
import collections.abc
from itertools import repeat
import numpy as np

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x,n))
    return parse
to_2tuple = _ntuple(2)


class ISA(nn.Module):
    def __init__(self,in_chans,num_heads,fd_k=3,fd_s=1,fd_p=1,attn_drop=0.,proj_drop=0.,qk_scale=None) -> None:
        super().__init__()
        #### suppose the out_chans = in_chans
        self.in_chans = in_chans
        self.out_chans = in_chans
        self.num_heads = num_heads
        head_dim = self.out_chans//num_heads
        self.stride=fd_s
        self.k = fd_k
        self.avgpl2d = nn.AvgPool2d(fd_k,fd_s,fd_p)
        self.qkv_chans = nn.Linear(in_chans,self.out_chans*4)
        self.unfold = nn.Unfold(kernel_size=fd_k,stride=fd_s,padding=fd_p)
        self.scale = qk_scale or head_dim**-0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.chans_attn = nn.Sequential(
            nn.Linear(self.out_chans,self.out_chans//2),
            nn.ReLU(),
            nn.Linear(self.out_chans//2,self.out_chans),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(self.out_chans,self.out_chans)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        #### suppose the input is [B,C,H,W]
        B, C, H, W = x.shape
        h, w = H//self.stride, W//self.stride
        x = self.avgpl2d(x).permute(0,2,3,1) #### [B,H,W,C]
        x = self.qkv_chans(x).reshape(B,h,w,C,4).permute(4,0,1,2,3) #### [3,B,H,W,C]
        q,k,v,chans = x[0],x[1],x[2],x[3] ### each one is [B,h,w,C]

        ### channel attention
        chs_attn = self.chans_attn(chans)

        ### manipulate q
        q = q.unsqueeze(4).reshape(B,h*w,C,1).reshape(B,h*w,self.num_heads,C//self.num_heads,1).permute(0,2,1,4,3) #### [B,nH,L,1,C//nH]
        
        ### manipulate k
        
        k = self.unfold(k.permute(0,3,1,2)).reshape(B,self.num_heads,C//self.num_heads,self.k**2,h*w).permute(0,1,4,2,3) ###(B,nH,L, C//nH,K*K)
        attn = (q @ k)*self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) ###(B,nH,L,1,k*k)

        ### manipulate v
        v = self.unfold(v.permute(0,3,1,2)).reshape(B,self.num_heads,C//self.num_heads,self.k**2,h*w).permute(0,1,4,3,2) ###(B,nH,L,K*K, C//nH)
        x = (attn @v).squeeze(3).permute(0,2,1,3).reshape(B,h,w,C)
        x = x*chs_attn + x
        x = self.proj(x) #### [B,h,w,C]
        x = self.proj_drop(x).permute(0,3,1,2)

        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From timm.layers.drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,im_height=None, im_width=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
            act_layer(),
        )
        self.proj_conv = nn.Sequential(nn.Conv2d(hidden_features, hidden_features, 1, 1, 0, groups=hidden_features),
                                       nn.BatchNorm2d(hidden_features, eps=1e-5),
                                       act_layer(),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = DropPath(drop)

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.drop(x0)
        x0 = self.proj_conv(x0) 
        x0 = self.conv2(x0)
        x0 = self.drop(x0)
        x = x + x0
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1]//patch_size[1]) * (img_size[0]//patch_size[0])
        
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        x = x.transpose(1, 2).reshape(B,self.embed_dim,H,W).contiguous()
        return x

class Classifier(nn.Module):
    """
    suppose the input is a 4D tensor as (B,C,H,W)
    output_shape: (B,num_class)
    """
    def __init__(self,in_chans, num_class) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.cls_conv = nn.Sequential(nn.Conv2d(in_chans, in_chans, 1, 1, 0),
                                      nn.BatchNorm2d(in_chans, eps=1e-5),
                                      nn.ReLU(),)
        self.cls_linear = nn.Linear(in_chans,num_class) if num_class >0 else nn.Identity()
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.cls_conv(x)
        x = x.view(x.size(0),-1)
        x = self.cls_linear(x)
        return x

class StemConv(nn.Module):
    def __init__(self, in_channels, stem_channels=16, img_size=7,kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.img_size = img_size
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(),

            nn.Conv2d(stem_channels, stem_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.stem_conv(x)
        x = x[:,:,1:,1:] if x.shape[-1]>self.img_size else x ### (B,16,8,8)
        return x

class FeatLevelFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, attn_drop=0.1, batch_first=True):
        """
        embed_dim/num_heads: should be interger. 
        nn.MultiheadAttention already has a linear projection layer, but without proj dropout, residual connection and pose embedding.
        joint key learning.
        """
        super().__init__()
        self.joint_key = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim//4, 1, 1, 0),
            nn.BatchNorm2d(embed_dim//4, eps=1e-5),
            nn.ReLU(inplace=True),

            nn.Conv2d(embed_dim//4, embed_dim//4, 3, 1, 1),
            nn.BatchNorm2d(embed_dim//4, eps=1e-5),
            nn.ReLU(inplace=True),

            nn.Conv2d(embed_dim//4, embed_dim, 1, 1, 0),
            nn.BatchNorm2d(embed_dim, eps=1e-5),
            nn.ReLU(inplace=True)
        )
        self.opt_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_drop,batch_first=batch_first)
        self.sar_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_drop,batch_first=batch_first)
        self.opt_sar_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_drop,batch_first=batch_first)
        self.sar_opt_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=attn_drop,batch_first=batch_first)

    def forward(self, opt, sar):
        """
        The q, k, v suppose to be the same, and the shape is (B, Eq, embed_dim), which is (B, H*W, C)
        But as the input may have shape as (B,C,H,W), it should be reshape to (B, H*W, C)
        if the input.dim == 3, we suppose it has the shape of (B, H*W, C)

        The input of MultiheadAttention is Q,K,V
        The output shape is (B, Eq, embed_dim) if opt.dim == 3 else (B,C,H,W)
        """
        input_dim = opt.dim()
        key = self.joint_key(torch.cat([opt, sar], dim=1))
        if input_dim == 4:
            B, C, H, W = opt.shape
            opt = opt.flatten(2).transpose(1, 2).contiguous()
            sar = sar.flatten(2).transpose(1, 2).contiguous()
            key = key.flatten(2).transpose(1,2).contiguous()
        # elif opt.dim == 3:
        #     B, N, C = opt.shape
        #     H, W = int(np.sqrt(N)), int(np.sqrt(N))
        opt_mhsa, _ = self.opt_attn(opt, opt, opt)
        sar_mhsa, _ = self.sar_attn(sar, sar, sar)
        opt_mhca, _ = self.opt_sar_attn(opt, key, opt)
        sar_mhca,_ = self.sar_opt_attn(sar, key, sar)

        opt_atten_out = opt_mhsa + opt_mhca
        sar_atten_out = sar_mhsa + sar_mhca

        if input_dim == 4:
            opt_atten_out = opt_atten_out.transpose(1, 2).reshape(B, C, H, W).contiguous()
            sar_atten_out = sar_atten_out.transpose(1, 2).reshape(B, C, H, W).contiguous()

        return opt_atten_out, sar_atten_out

class FeatFuseBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, mlp_ratio=4.,drop=0.,attn_drop=0.1,drop_path=0.,act_layer=nn.GELU,norm_layer=nn.LayerNorm,batch_first=True):
        super().__init__()
        self.fuse_op = FeatLevelFusion(embed_dim, num_heads, attn_drop, batch_first)
        self.norm = norm_layer(embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, opt, sar):
        B,C,H,W = opt.shape
        shortcut_opt = opt
        shortcut_sar = sar
        opt, sar = opt.flatten(2).transpose(1, 2).contiguous(), sar.flatten(2).transpose(1, 2).contiguous()
        opt, sar = self.norm(opt), self.norm(sar)
        opt, sar = opt.transpose(1,2).reshape(B,C,H,W).contiguous(), sar.transpose(1,2).reshape(B,C,H,W).contiguous()
        opt, sar = self.fuse_op(opt, sar)
        opt, sar = self.drop_path(opt) + shortcut_opt, self.drop_path(sar) + shortcut_sar

        x = opt+sar
        shortcut = x
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1,2).reshape(B,C,H,W).contiguous()
        x = self.drop_path(self.mlp(x)) + shortcut
        return x
    
class DecisionLevelFusion(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.weight_conv = nn.Conv2d(dim,dim,1,1,0,groups=dim)
    def forward(self, opt, sar):
        concat = (opt+sar).view(opt.size(0),opt.size(1),1,1)
        weight = torch.sigmoid(self.weight_conv(concat)).squeeze(-1).squeeze(-1)
        return weight*(opt+sar)
    
class DecisionFuseBlock(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.fuse_op = DecisionLevelFusion(dim)

    def forward(self, opt, sar):
        return self.fuse_op(opt, sar)
    
class Block(nn.Module):
    """
    Block for transformer structure.
    input.dim==4 and output.dim==4
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,batch_first=False,use_isa=False):
        super().__init__()
        self.use_isa = use_isa
        self.dim = dim
        self.norm = norm_layer(dim)
        # self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=batch_first)
        self.attn = ISA(dim, num_heads, attn_drop=attn_drop, proj_drop=drop) if use_isa else nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=batch_first)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        """
        suppose the input.dim==4
        """
        B, _, H, W = x.shape
        shortcut = x
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        if self.use_isa:
            x = x.transpose(1,2).reshape(B,self.dim,H,W).contiguous()
            x = self.attn(x)
        else:
            x, _ = self.attn(x,x,x)
            x = x.transpose(1,2).reshape(B,self.dim,H,W).contiguous()
        x = self.drop_path(x) + shortcut

        shortcut = x
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        x = x.transpose(1,2).reshape(B,self.dim,H,W).contiguous()
        x = self.drop_path(self.mlp(x)) + shortcut

        return x

class BaseModel(nn.Module):
    ###### single branch for either opt or sar, input should be even 
    def __init__(self,img_size=7,in_chans=12,stem_chans=16,embed_dim=[24,48,96],num_heads=[4,8,16],mlp_ratio=4.,
                 depths=[2,8,2],drop=0.,attn_drop=0.05,drop_path_rate=0.1,act_layer=nn.GELU,norm_layer=nn.LayerNorm,
                 batch_first=True, use_isa=True):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        ### define operation

        self.pted_1 = PatchEmbed(img_size=img_size, patch_size=2,in_chans=stem_chans,embed_dim=embed_dim[0])
        self.block_1 = nn.ModuleList([
            Block(embed_dim[0], num_heads[0], mlp_ratio, drop, attn_drop, dpr[i], act_layer, norm_layer,batch_first,use_isa=use_isa)
            for i in range(depths[0])])
        
        self.pted_2 = PatchEmbed(img_size=img_size//2, patch_size=1,in_chans=embed_dim[0],embed_dim=embed_dim[1])
        self.block_2 = nn.ModuleList([
            Block(embed_dim[1], num_heads[1], mlp_ratio, drop, attn_drop, dpr[i+sum(depths[:1])], act_layer, norm_layer,batch_first)
            for i in range(depths[1])])
        
        self.pted_3 = PatchEmbed(img_size=img_size//2, patch_size=2,in_chans=embed_dim[1],embed_dim=embed_dim[2])
        self.block_3 = nn.ModuleList([
            Block(embed_dim[2], num_heads[2], mlp_ratio, drop, attn_drop, dpr[i+sum(depths[:2])], act_layer, norm_layer, batch_first)
            for i in range(depths[2])])
    
    def forward(self, x):

        x = self.pted_1(x) ### (B,24,4,4)
        for block in self.block_1:
            x = block(x)
        # x = self.block_1(x) ### 4x4x24

        x = self.pted_2(x) ### 4x4x48
        for block in self.block_2:
            x = block(x)
        # x = self.block_2(x) ### 4x4x48

        x = self.pted_3(x) ### 2x2x96
        for block in self.block_3:
            x = block(x)
        # x = self.block_3(x) ### 2x2x96
        return x

class SoftFormer(nn.Module):
    def __init__(self,img_size=8,opt_chans=12,sar_chans=10,num_class=6,stem_chans=16,embed_dim=[24,48,96],num_heads=[4,8,16],mlp_ratio=4.,
                 depths=[2,8,2],drop=0.,attn_drop=0.1,drop_path_rate=0.1,act_layer=nn.GELU,norm_layer=nn.LayerNorm,
                 batch_first=True,use_isa=True):
        super().__init__()
        self.opt_stem, self.sar_stem = StemConv(opt_chans,stem_chans,img_size), StemConv(sar_chans,stem_chans,img_size)
        self.opt_encoder = BaseModel(img_size,stem_chans,stem_chans,embed_dim,num_heads,mlp_ratio,
                                     depths,drop,attn_drop,drop_path_rate,act_layer,norm_layer,batch_first,use_isa)
        self.sar_encoder = BaseModel(img_size,stem_chans,stem_chans,embed_dim,num_heads,mlp_ratio,
                                     depths,drop,attn_drop,drop_path_rate,act_layer,norm_layer,batch_first,use_isa)
        self.feat_fusion = FeatFuseBlock(embed_dim[-1],num_heads[-1],mlp_ratio,drop,attn_drop,
                                         drop_path_rate,act_layer,norm_layer,batch_first)
        self.cls_opt = Classifier(embed_dim[-1], num_class)
        self.cls_sar = Classifier(embed_dim[-1], num_class)
        self.cls_fut = Classifier(embed_dim[-1], num_class)
        self.dcs_fu_optsar = DecisionFuseBlock(num_class)
        self.dcs_fu_featfu = DecisionFuseBlock(num_class)
    
    def forward(self, opt, sar):
        
        opt, sar = self.opt_stem(opt), self.sar_stem(sar)
        opt, sar = self.opt_encoder(opt), self.sar_encoder(sar)
        x = self.feat_fusion(opt,sar)

        opt, sar = self.cls_opt(opt), self.cls_sar(sar)
        x = self.cls_fut(x)

        pred_fu = self.dcs_fu_optsar(opt,sar)
        pred_logits = self.dcs_fu_featfu(pred_fu,x)

        return (pred_logits, pred_fu, x, opt, sar)
