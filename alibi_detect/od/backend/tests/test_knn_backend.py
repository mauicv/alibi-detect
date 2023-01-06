import pytest
import torch

from alibi_detect.od.backend.torch.knn import KNNTorch
from alibi_detect.utils.pytorch.kernels import GaussianRBF
from alibi_detect.od.backend.torch.ensemble import Accumulator, PValNormalizer, AverageAggregator
from alibi_detect.od.base import NotFitException, ThresholdNotInferredException


@pytest.fixture(scope='session')
def accumulator(request):
    return Accumulator(
        normalizer=PValNormalizer(),
        aggregator=AverageAggregator()
    )


def test_knn_torch_backend():
    knn_torch = KNNTorch(k=5)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    outputs = knn_torch.predict(x)
    assert outputs.instance_score.shape == (3, )
    assert outputs.is_outlier is None
    assert outputs.p_value is None
    scores = knn_torch.score(x)
    assert torch.all(scores == outputs.instance_score)

    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(knn_torch(x) == torch.tensor([False, False, True]))


def test_knn_torch_backend_ensemble(accumulator):
    knn_torch = KNNTorch(k=[4, 5], accumulator=accumulator)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    result = knn_torch.predict(x)
    assert result.instance_score.shape == (3, )

    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(knn_torch(x) == torch.tensor([False, False, True]))


def test_knn_torch_backend_ensemble_ts(tmp_path, accumulator):
    knn_torch = KNNTorch(k=[4, 5], accumulator=accumulator)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])

    with pytest.raises(NotFitException) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    with pytest.raises(NotFitException) as err:
        knn_torch.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    knn_torch.infer_threshold(x_ref, 0.1)
    pred_1 = knn_torch(x)
    knn_torch = torch.jit.script(knn_torch)
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)

    knn_torch.save(tmp_path / 'knn_torch.pt')
    knn_torch = torch.load(tmp_path / 'knn_torch.pt')
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_knn_torch_backend_ts(tmp_path):
    knn_torch = KNNTorch(k=7)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    knn_torch.infer_threshold(x_ref, 0.1)
    pred_1 = knn_torch(x)
    knn_torch = torch.jit.script(knn_torch)
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)

    knn_torch.save(tmp_path / 'knn_torch.pt')
    knn_torch = torch.load(tmp_path / 'knn_torch.pt')
    pred_2 = knn_torch(x)
    assert torch.all(pred_1 == pred_2)


def test_knn_kernel(accumulator):
    kernel = GaussianRBF(sigma=torch.tensor((0.25)))
    knn_torch = KNNTorch(k=[4, 5], kernel=kernel, accumulator=accumulator)
    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    x = torch.randn((3, 10)) * torch.tensor([[1], [1], [100]])
    result = knn_torch.predict(x)
    assert result.instance_score.shape == (3,)

    knn_torch.infer_threshold(x_ref, 0.1)
    outputs = knn_torch.predict(x)
    assert torch.all(outputs.is_outlier == torch.tensor([False, False, True]))
    assert torch.all(knn_torch(x) == torch.tensor([False, False, True]))

    """Can't convert GaussianRBF to torchscript due to torchscript type
    constraints"""
    # pred_1 = knn_torch(x)
    # knn_torch = torch.jit.script(knn_torch)
    # pred_2 = knn_torch(x)
    # assert torch.all(pred_1 == pred_2)


def test_knn_torch_backend_ensemble_fit_errors(accumulator):
    knn_torch = KNNTorch(k=[4, 5], accumulator=accumulator)
    assert not knn_torch._fitted

    x = torch.randn((1, 10))
    with pytest.raises(NotFitException) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    with pytest.raises(NotFitException) as err:
        knn_torch.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)
    assert knn_torch._fitted

    with pytest.raises(ThresholdNotInferredException) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has no threshold set, call `infer_threshold` before predicting.'

    assert knn_torch.predict(x)


def test_knn_torch_backend_fit_errors():
    knn_torch = KNNTorch(k=4)
    assert not knn_torch._fitted

    x = torch.randn((1, 10))
    with pytest.raises(NotFitException) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    with pytest.raises(NotFitException) as err:
        knn_torch.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'

    x_ref = torch.randn((1024, 10))
    knn_torch.fit(x_ref)

    assert knn_torch._fitted

    with pytest.raises(ThresholdNotInferredException) as err:
        knn_torch(x)
    assert str(err.value) == 'KNNTorch has no threshold set, call `infer_threshold` before predicting.'

    assert knn_torch.predict(x)
