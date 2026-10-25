"""Compact Fashion-MNIST networks; all inputs are NCHW grayscale images."""
from __future__ import annotations

import torch
from torch import nn

from .constants import DEFAULT_IMAGE_SIZE, DEFAULT_INPUT_CHANNELS, DEFAULT_NUM_CLASSES


def _pair(value):
    pair=(value,value) if isinstance(value,int) else tuple(value)
    if len(pair)!=2 or any(dimension<=0 for dimension in pair):
        raise ValueError("image_size must be a positive integer or a pair of positive integers")
    return pair

class MLP(nn.Module):
    def __init__(self, hidden_dims=(256,128), dropout: float=0., num_classes: int=DEFAULT_NUM_CLASSES,
                 image_size: int | tuple[int, int]=DEFAULT_IMAGE_SIZE, input_channels: int=DEFAULT_INPUT_CHANNELS):
        super().__init__(); layers=[]; image_height,image_width=_pair(image_size); size=input_channels*image_height*image_width
        for width in hidden_dims:
            layers += [nn.Linear(size,width),nn.ReLU()]
            if dropout: layers.append(nn.Dropout(dropout))
            size=width
        layers += [nn.Linear(size,num_classes)]; self.net=nn.Sequential(nn.Flatten(),*layers)
    def forward(self,x): return self.net(x)

class FashionCNN(nn.Module):
    def __init__(self, channels=(32,64), dropout: float=.1, num_classes: int=DEFAULT_NUM_CLASSES,
                 image_size: int | tuple[int, int]=DEFAULT_IMAGE_SIZE, input_channels: int=DEFAULT_INPUT_CHANNELS,
                 classifier_dim: int=128):
        super().__init__(); blocks=[]; incoming=input_channels; height,width=_pair(image_size)
        for c in channels: blocks += [nn.Conv2d(incoming,c,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)]; incoming=c
        self.features=nn.Sequential(*blocks)
        for _ in channels: height,width=height//2,width//2
        if height < 1 or width < 1: raise ValueError("image_size is too small for the configured pooling layers")
        self.image_size=_pair(image_size); self.input_channels=input_channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(incoming * height * width, classifier_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )
    def forward(self,x):
        if x.shape[1] != self.input_channels or tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(f"expected inputs shaped [batch, {self.input_channels}, {self.image_size[0]}, {self.image_size[1]}]")
        return self.classifier(self.features(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__(); self.norm1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True);
        self.norm2=nn.LayerNorm(dim);
        self.ff=nn.Sequential(nn.Linear(dim,mlp_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(mlp_dim,dim));
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        # x [batch, patches+1, embedding]; attention weights [batch, heads, query, key].
        z=self.norm1(x); a,_=self.attn(z,z,z,need_weights=False); x=x+self.drop(a); return x+self.drop(self.ff(self.norm2(x)))

class TinyViT(nn.Module):
    def __init__(self, patch_size: int=7, dim: int=64, depth: int=2, heads: int=4, mlp_dim: int=128,
                 dropout: float=.1, num_classes: int=DEFAULT_NUM_CLASSES,
                 image_size: int | tuple[int, int]=DEFAULT_IMAGE_SIZE, input_channels: int=DEFAULT_INPUT_CHANNELS):
        super().__init__()
        height,width=_pair(image_size)
        if input_channels <= 0 or heads <= 0 or patch_size <= 0 or height%patch_size or width%patch_size or dim%heads:
            raise ValueError("patch_size must divide each image dimension, and dim must divide heads")
        self.patch_size=patch_size; self.image_size=(height,width); self.input_channels=input_channels
        n=(height//patch_size)*(width//patch_size)
        self.embed=nn.Linear(input_channels*patch_size*patch_size,dim); self.cls=nn.Parameter(torch.zeros(1,1,dim));
        self.pos=nn.Parameter(torch.zeros(1,n+1,dim));
        self.blocks=nn.Sequential(*[TransformerBlock(dim,heads,mlp_dim,dropout) for _ in range(depth)]);
        self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,num_classes)
    def forward(self,x):
        b,c,h,w=x.shape; p=self.patch_size
        if (h,w)!=self.image_size or c!=self.input_channels:
            raise ValueError(f"expected inputs shaped [batch, {self.input_channels}, {self.image_size[0]}, {self.image_size[1]}]")
        # Unfold gives [batch, channels*patch_area, number_of_patches], then transpose to [batch, patches, patch_area].
        patches=x.unfold(2,p,p).unfold(3,p,p).permute(0,2,3,1,4,5).contiguous().view(b,-1,c*p*p)
        tokens=self.embed(patches); tokens=torch.cat([self.cls.expand(b,-1,-1),tokens],dim=1)+self.pos
        return self.head(self.norm(self.blocks(tokens)[:,0]))
