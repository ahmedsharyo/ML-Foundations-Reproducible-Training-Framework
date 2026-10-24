import pytest
torch=pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ml_foundations.training.engine import fit,load_checkpoint,evaluate
from ml_foundations.utils.reproducibility import get_device
from ml_foundations.data.splits import deterministic_split

def test_device_and_split():
    assert get_device("cpu").type=="cpu"; assert deterministic_split(20,.2,9)==deterministic_split(20,.2,9)

def test_fit_and_checkpoint_resume(tmp_path):
    data=TensorDataset(torch.randn(12,4),torch.randint(0,2,(12,))); loader=DataLoader(data,batch_size=4); model=nn.Linear(4,2); opt=torch.optim.SGD(model.parameters(),lr=.1); path=tmp_path/"best.pt"
    fit(model,loader,loader,opt,nn.CrossEntropyLoss(),"cpu",1,path,{"model":{"type":"test"}}); state=load_checkpoint(path,model,opt); assert state["epoch"]==1
    assert "accuracy" in evaluate(model,loader,nn.CrossEntropyLoss(),"cpu")
