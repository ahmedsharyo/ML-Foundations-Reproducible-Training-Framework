import pytest
torch=pytest.importorskip("torch")
from ml_foundations.models.torch_models import MLP,FashionCNN,TinyViT
from ml_foundations.models.constants import DEFAULT_IMAGE_SIZE, DEFAULT_INPUT_CHANNELS, DEFAULT_NUM_CLASSES

@pytest.mark.parametrize("model",[MLP(hidden_dims=(16,)),FashionCNN(channels=(4,)),TinyViT(dim=16,heads=4,depth=1,mlp_dim=32)])
def test_forward_backward(model):
    x=torch.randn(2,DEFAULT_INPUT_CHANNELS,DEFAULT_IMAGE_SIZE,DEFAULT_IMAGE_SIZE); y=model(x); assert y.shape==(2,DEFAULT_NUM_CLASSES); y.sum().backward()
