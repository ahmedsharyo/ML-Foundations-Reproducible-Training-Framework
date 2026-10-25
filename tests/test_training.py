import pytest
torch=pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from ml_foundations.training.engine import fit,load_checkpoint,evaluate
from ml_foundations.utils.reproducibility import get_device
from ml_foundations.data.splits import deterministic_split

NUM_SAMPLES = 12
INPUT_DIM = 4
NUM_CLASSES = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.1
EPOCHS = 1


def test_device_and_split():
    assert get_device("cpu").type=="cpu";
    assert deterministic_split(20,.2,9)==deterministic_split(20,.2,9)

def test_fit_and_checkpoint_resume(tmp_path):
    data = TensorDataset(torch.randn(NUM_SAMPLES, INPUT_DIM), torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,)))
    loader = DataLoader(data, batch_size=BATCH_SIZE)
    model = nn.Linear(INPUT_DIM, NUM_CLASSES)
    opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    path = tmp_path / "best.pt"
    fit(model, loader, loader, opt, nn.CrossEntropyLoss(), "cpu", EPOCHS, path, {"model": {"type": "test"}})
    state = load_checkpoint(path, model, opt)
    assert state["epoch"] == EPOCHS
    assert "accuracy" in evaluate(model, loader, nn.CrossEntropyLoss(), "cpu")

