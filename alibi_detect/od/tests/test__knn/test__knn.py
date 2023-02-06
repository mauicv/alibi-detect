import pytest
import numpy as np
import torch

from alibi_detect.od._knn import KNN
from alibi_detect.od import AverageAggregator, TopKAggregator, MaxAggregator, \
    MinAggregator, ShiftAndScaleNormalizer, PValNormalizer
from alibi_detect.base import NotFitException

from sklearn.datasets import make_moons


def make_knn_detector(k=5, aggregator=None, normalizer=None):
    knn_detector = KNN(
        k=k, aggregator=aggregator,
        normalizer=normalizer
    )
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    knn_detector.infer_threshold(x_ref, 0.1)
    return knn_detector


def test_unfitted_knn_single_score():
    knn_detector = KNN(k=10)
    x = np.array([[0, 10], [0.1, 0]])

    # test predict raises exception when not fitted
    with pytest.raises(NotFitException) as err:
        _ = knn_detector.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'


@pytest.mark.parametrize('k', [10, [8, 9, 10]])
def test_fitted_knn_single_score(k):
    knn_detector = KNN(k=k)
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0.1, 0]])

    # test fitted but not threshold inferred detectors
    # can still score data using the predict method.
    y = knn_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 1

    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


def test_incorrect_knn_ensemble_init():
    # test knn ensemble with aggregator passed as None raises exception

    with pytest.raises(ValueError) as err:
        KNN(k=[8, 9, 10], aggregator=None)
    assert str(err.value) == ('If `k` is a `np.ndarray`, `list` or `tuple`, '
                              'the `aggregator` argument cannot be ``None``.')


def test_fitted_knn_predict():
    knn_detector = make_knn_detector(k=10)
    x_ref = np.random.randn(100, 2)

    # test detector fitted on data and with threshold inferred correctly scores and
    # labels outliers, as well as return the p-values using the predict method.
    knn_detector.infer_threshold(x_ref, 0.1)
    x = np.array([[0, 10], [0, 0.1]])
    y = knn_detector.predict(x)
    y = y['data']
    assert y['instance_score'][0] > 5
    assert y['instance_score'][1] < 1
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None])
def test_unfitted_knn_ensemble(aggregator, normalizer):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x = np.array([[0, 10], [0.1, 0]])

    # Test unfit knn ensemble raises exception when calling predict method.
    with pytest.raises(NotFitException) as err:
        _ = knn_detector.predict(x)
    assert str(err.value) == 'KNNTorch has not been fit!'


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None])
def test_fitted_knn_ensemble(aggregator, normalizer):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])

    # test fitted but not threshold inferred detectors can still score data using the predict method.
    y = knn_detector.predict(x)
    y = y['data']
    assert y['instance_score'].all()
    assert not y['threshold_inferred']
    assert y['threshold'] is None
    assert y['is_outlier'] is None
    assert y['p_value'] is None


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None])
def test_fitted_knn_ensemble_predict(aggregator, normalizer):
    knn_detector = make_knn_detector(
        k=[8, 9, 10],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    x = np.array([[0, 10], [0, 0.1]])

    # test fitted detectors with inferred thresholds can score data using the predict method.
    y = knn_detector.predict(x)
    y = y['data']
    assert y['threshold_inferred']
    assert y['threshold'] is not None
    assert y['p_value'].all()
    assert (y['is_outlier'] == [True, False]).all()


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None])
def test_knn_ensemble_torch_script(aggregator, normalizer):
    knn_detector = make_knn_detector(k=[5, 6, 7], aggregator=aggregator(), normalizer=normalizer())
    tsknn = torch.jit.script(knn_detector.backend)
    x = torch.tensor([[0, 10], [0, 0.1]])

    # test torchscripted ensemble knn detector can be saved and loaded correctly.
    y = tsknn(x)
    assert torch.all(y == torch.tensor([True, False]))


def test_knn_single_torchscript():
    knn_detector = make_knn_detector(k=5)
    tsknn = torch.jit.script(knn_detector.backend)
    x = torch.tensor([[0, 10], [0, 0.1]])

    # test torchscripted single knn detector can be saved and loaded correctly.
    y = tsknn(x)
    assert torch.all(y == torch.tensor([True, False]))


@pytest.mark.parametrize("aggregator", [AverageAggregator, lambda: TopKAggregator(k=7),
                                        MaxAggregator, MinAggregator, lambda: 'AverageAggregator',
                                        lambda: 'TopKAggregator', lambda: 'MaxAggregator',
                                        lambda: 'MinAggregator'])
@pytest.mark.parametrize("normalizer", [ShiftAndScaleNormalizer, PValNormalizer, lambda: None,
                                        lambda: 'ShiftAndScaleNormalizer', lambda: 'PValNormalizer'])
def test_knn_ensemble_integration(aggregator, normalizer):
    """Test knn ensemble detector on moons dataset.

    Tests ensemble knn detector with every combination of aggregator and normalizer on the moons dataset.
    Fits and infers thresholds in each case. Verifies that the detector can correctly detect inliers
    and outliers and that it can be serialized using the torchscript.
    """

    knn_detector = KNN(
        k=[10, 14, 18],
        aggregator=aggregator(),
        normalizer=normalizer()
    )
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    knn_detector.fit(X_ref)
    knn_detector.infer_threshold(X_ref, 0.1)
    result = knn_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = knn_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result

    tsknn = torch.jit.script(knn_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = tsknn(x)
    assert torch.all(y == torch.tensor([False, True]))


def test_knn_integration():
    """Test knn detector on moons dataset.

    Tests knn detector on the moons dataset. Fits and infers thresholds and verifies that the detector can
    correctly detect inliers and outliers. Checks that it can be serialized using the torchscript.
    """
    knn_detector = KNN(k=18)
    X_ref, _ = make_moons(1001, shuffle=True, noise=0.05, random_state=None)
    X_ref, x_inlier = X_ref[0:1000], X_ref[1000][None]
    knn_detector.fit(X_ref)
    knn_detector.infer_threshold(X_ref, 0.1)
    result = knn_detector.predict(x_inlier)
    result = result['data']['is_outlier'][0]
    assert not result

    x_outlier = np.array([[-1, 1.5]])
    result = knn_detector.predict(x_outlier)
    result = result['data']['is_outlier'][0]
    assert result

    tsknn = torch.jit.script(knn_detector.backend)
    x = torch.tensor([x_inlier[0], x_outlier[0]], dtype=torch.float32)
    y = tsknn(x)
    assert torch.all(y == torch.tensor([False, True]))
