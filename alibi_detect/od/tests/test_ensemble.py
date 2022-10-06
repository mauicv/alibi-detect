import numpy as np
from alibi_detect.od.config import write_config
from alibi_detect.od.loading import load_detector
from alibi_detect.od.ensemble import Ensemble
from alibi_detect.od.knn import KNN
from alibi_detect.od.aggregation import AverageAggregator, PValNormaliser
from alibi_detect.od.processor import ParallelProcessor


def test_ensemble():
    knn_detectors = [KNN(k=k+1) for k in range(10)]
    ensemble_detector = Ensemble(
        detectors=knn_detectors, 
        aggregator=AverageAggregator(), 
        normaliser=PValNormaliser()
    )
    x_ref = np.random.randn(100, 2)
    ensemble_detector.fit(x_ref)
    x = np.array([[0, 0.1]])
    ensemble_detector.infer_threshold(x_ref, 0.1)
    assert ensemble_detector.predict(x)['p_vals'].item() > 0.5


def test_ensemble_parrallel():
    knn_detectors = [KNN(k=k+1) for k in range(10)]
    ensemble_detector = Ensemble(
        detectors=knn_detectors, 
        aggregator=AverageAggregator(), 
        normaliser=PValNormaliser(),
        processor=ParallelProcessor()
    )
    x_ref = np.random.randn(100, 2)
    ensemble_detector.fit(x_ref)
    x = np.array([[0, 0.1]])
    ensemble_detector.infer_threshold(x_ref, 0.1)
    assert ensemble_detector.predict(x)['p_vals'].item() > 0.5


def test_ensemble_config(tmp_path):
    knn_detectors = [KNN(k=k+1) for k in range(10)]
    ensemble_detector = Ensemble(
        detectors=knn_detectors,
        aggregator=AverageAggregator(),
        normaliser=PValNormaliser(),
        processor=ParallelProcessor()
    )

    config = ensemble_detector.save(tmp_path)
    loaded_ensemble_detector = load_detector(tmp_path)

    assert isinstance(loaded_ensemble_detector, Ensemble)
    assert isinstance(loaded_ensemble_detector.aggregator, AverageAggregator)
    assert isinstance(loaded_ensemble_detector.normaliser, PValNormaliser)
    assert isinstance(loaded_ensemble_detector.processor, ParallelProcessor)
    for ind, detector in enumerate(loaded_ensemble_detector.detectors):
        assert isinstance(detector, KNN)
        detector.k = ind + 1