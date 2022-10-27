from re import L
from alibi_detect.od.deepsvdd import DeepSVDD
from alibi_detect.od.backends import DeepSVDDTorch
import torch
from alibi_detect.od.loading import load_detector
from alibi_detect.od.config import ModelWrapper
from torch.utils.data import DataLoader
from alibi_detect.utils.pytorch.data import TorchDataset
# from alibi_detect.od.backends.torch import DeepSVDDTorch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc1(x)


def test_deepsvdd_config(tmp_path):
    model = Model()
    deepsvdd_detector = DeepSVDD(model)
    path = deepsvdd_detector.save(tmp_path)
    loaded_deepsvdd_detector = load_detector(path)
    assert isinstance(loaded_deepsvdd_detector, DeepSVDD)
    assert loaded_deepsvdd_detector.backend.__class__.__name__ == DeepSVDDTorch.__name__
    assert loaded_deepsvdd_detector.backend.dataloader.func == DataLoader
    assert loaded_deepsvdd_detector.backend.dataset == TorchDataset
    assert isinstance(loaded_deepsvdd_detector.model, Model)
    assert isinstance(loaded_deepsvdd_detector.original_model, ModelWrapper)


# def test_deepsvdd_backend():
#     DeepSVDDTorch()


def test_deepsvdd(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 2, bias=False)
            self.fc2 = nn.Linear(2, 2, bias=False)
            self.fc3 = nn.Linear(2, 1, bias=False)

        def forward(self, x):
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            return self.fc3(x)

    model = Model()
    deepsvdd_detector = DeepSVDD(model, verbose=0)
    x_ref = np.random.randn(100, 2)
    deepsvdd_detector.fit(x_ref)

    x = np.array([[0, 10]])
    print(deepsvdd_detector.score(x))

    x = np.array([[0.4, 0.1]])
    print(deepsvdd_detector.score(x))
