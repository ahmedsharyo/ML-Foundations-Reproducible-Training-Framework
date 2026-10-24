"""Compact Fashion-MNIST networks; all inputs are NCHW grayscale images."""
from __future__ import annotations
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden_dims=(256,128), dropout: float=0., num_classes: int=10):
        super().__init__(); layers=[]; size=28*28
        for width in hidden_dims:
            layers += [nn.Linear(size,width),nn.ReLU()]
            if dropout: layers.append(nn.Dropout(dropout))
            size=width
        layers += [nn.Linear(size,num_classes)]; self.net=nn.Sequential(nn.Flatten(),*layers)
    def forward(self,x): return self.net(x)

class FashionCNN(nn.Module):
    def __init__(self, channels=(32,64), dropout: float=.1, num_classes: int=10):
        super().__init__(); blocks=[]; incoming=1
        for c in channels: blocks += [nn.Conv2d(incoming,c,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)]; incoming=c
        self.features=nn.Sequential(*blocks); side=28//(2**len(channels)); self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(incoming*side*side,128),nn.ReLU(),nn.Dropout(dropout),nn.Linear(128,num_classes))
    def forward(self,x): return self.classifier(self.features(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__(); self.norm1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True); self.norm2=nn.LayerNorm(dim); self.ff=nn.Sequential(nn.Linear(dim,mlp_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(mlp_dim,dim)); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        # x [batch, patches+1, embedding]; attention weights [batch, heads, query, key].
        z=self.norm1(x); a,_=self.attn(z,z,z,need_weights=False); x=x+self.drop(a); return x+self.drop(self.ff(self.norm2(x)))

class TinyViT(nn.Module):
    def __init__(self, patch_size: int=7, dim: int=64, depth: int=2, heads: int=4, mlp_dim: int=128, dropout: float=.1, num_classes: int=10):
        super().__init__()
        if 28%patch_size or dim%heads: raise ValueError("patch_size must divide 28 and dim must divide heads")
        self.patch_size=patch_size; n=(28//patch_size)**2
        self.embed=nn.Linear(patch_size*patch_size,dim); self.cls=nn.Parameter(torch.zeros(1,1,dim)); self.pos=nn.Parameter(torch.zeros(1,n+1,dim)); self.blocks=nn.Sequential(*[TransformerBlock(dim,heads,mlp_dim,dropout) for _ in range(depth)]); self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,num_classes)
    def forward(self,x):
        b,c,h,w=x.shape; p=self.patch_size
        # Unfold gives [batch, channels*patch_area, number_of_patches], then transpose to [batch, patches, patch_area].
        patches=x.unfold(2,p,p).unfold(3,p,p).contiguous().view(b,c,-1,p*p).squeeze(1)
        tokens=self.embed(patches); tokens=torch.cat([self.cls.expand(b,-1,-1),tokens],dim=1)+self.pos
        return self.head(self.norm(self.blocks(tokens)[:,0]))
