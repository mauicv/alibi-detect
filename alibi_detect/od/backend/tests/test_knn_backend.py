import torch

from alibi_detect.od.backend.torch.knn import KNNTorch
from alibi_detect.utils.pytorch.kernels import GaussianRBF


def test_knn_torch_backend(accumulator):
    knn_torch = KNNTorch(k=5, accumulator=None)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10))
    scores = knn_torch(x)
    assert scores.shape == (3, )
    knn_torch = torch.jit.script(knn_torch)
    scores_2 = knn_torch(x)
    assert torch.all(scores == scores_2)


def test_knn_torch_backend_ensemble(accumulator):
    knn_torch = KNNTorch(k=[4, 5], accumulator=accumulator)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10))
    scores = knn_torch(x)
    assert scores.shape == (3,)
    knn_torch = torch.jit.script(knn_torch)
    scores_2 = knn_torch(x)
    assert torch.all(scores == scores_2)


def test_knn_kernel(accumulator):
    kernel = GaussianRBF(sigma=torch.tensor((0.1)))
    knn_torch = KNNTorch(k=[4, 5], kernel=kernel, accumulator=accumulator)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10))
    scores = knn_torch(x)
    assert scores.shape == (3,)
    # knn_torch = torch.jit.script(knn_torch)
    # scores_2 = knn_torch(x)
    # assert torch.all(scores == scores_2)
