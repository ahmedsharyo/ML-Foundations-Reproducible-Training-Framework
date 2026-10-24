import pytest
torch=pytest.importorskip("torch")
from ml_foundations.models.torch_models import MLP,FashionCNN,TinyViT

@pytest.mark.parametrize("model",[MLP(hidden_dims=(16,)),FashionCNN(channels=(4,)),TinyViT(dim=16,heads=4,depth=1,mlp_dim=32)])
def test_forward_backward(model):
    x=torch.randn(2,1,28,28); y=model(x); assert y.shape==(2,10); y.sum().backward()
