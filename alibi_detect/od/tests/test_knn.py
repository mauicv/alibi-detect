import numpy as np
import torch
import os

from alibi_detect.od.knn import KNN
from alibi_detect.od.ensemble import Ensemble
from alibi_detect.od.loading import load_detector
from alibi_detect.od.aggregation import AverageAggregator, ShiftAndScaleNormaliser, PValNormaliser, TopKAggregator
from alibi_detect.od.backends import KNNTorch
from alibi_detect.od.backends.torch.saving import save_to_torch_script
from alibi_detect.od.aggregation import Transform


def test_knn_single():
    knn_detector = KNN(k=10)
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10]])
    assert knn_detector.predict(x)['raw_scores'] > 5

    x = np.array([[0, 0.1]])
    assert knn_detector.predict(x)['raw_scores'] < 1

    knn_detector.infer_threshold(x_ref, 0.1)

    x = np.array([[0, 10]])
    pred = knn_detector.predict(x)
    assert pred['raw_scores'] > 5
    assert pred['preds']
    assert pred['p_vals'] < 0.05

    x = np.array([[0, 0.1]])
    pred = knn_detector.predict(x)
    assert pred['raw_scores'] < 1
    assert not pred['preds']
    assert pred['p_vals'] > 0.7


def test_knn_ensemble():
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=AverageAggregator(),
        normaliser=ShiftAndScaleNormaliser()
    )

    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    knn_detector.infer_threshold(x_ref, 0.1)
    pred = knn_detector.predict(x)

    assert (pred['normalised_scores'][0].numpy() > 1).all()
    assert (pred['normalised_scores'][1].numpy() < 0).all()

    assert (pred['preds'].numpy() == np.array([True, False])).all()

    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=AverageAggregator(),
        normaliser=PValNormaliser()
    )

    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    x = np.array([[0, 10], [0, 0.1]])
    knn_detector.infer_threshold(x_ref, 0.1)
    pred = knn_detector.predict(x)

    assert np.all(pred['normalised_scores'][0].numpy() > 0.8)
    assert np.all(pred['normalised_scores'][1].numpy() < 0.3)
    assert (pred['preds'].numpy() == [True, False]).all()


# def test_knn_keops():
#     knn_detector = KNN(
#         k=[8, 9, 10],
#         aggregator=AverageAggregator(),
#         normaliser=ShiftAndScaleNormaliser(),
#         backend='keops'
#     )

#     x_ref = np.random.randn(100, 2)
#     knn_detector.fit(x_ref)
#     x = np.array([[0, 10], [0, 0.1]])
#     knn_detector.infer_threshold(x_ref, 0.1)
#     pred = knn_detector.predict(x)

#     assert np.all(pred['normalised_scores'][0] > 1)
#     assert np.all(pred['normalised_scores'][1] < 0) # Is this correct?
#     assert np.all(pred['preds'] == [True, False])

#     knn_detector = KNN(
#         k=[8, 9, 10],
#         aggregator=AverageAggregator(),
#         normaliser=PValNormaliser()
#     )

#     x_ref = np.random.randn(100, 2)
#     knn_detector.fit(x_ref)
#     x = np.array([[0, 10], [0, 0.1]])
#     knn_detector.infer_threshold(x_ref, 0.1)
#     pred = knn_detector.predict(x)

#     assert np.all(pred['normalised_scores'][0] > 0.8)
#     assert np.all(pred['normalised_scores'][1] < 0.3)
#     assert np.all(pred['preds'] == [True, False])


def test_knn_config(tmp_path):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=TopKAggregator(k=5),
        normaliser=ShiftAndScaleNormaliser(),
        backend='pytorch',
        # kernel=GaussianRBF(
        #     np.array([1.]),
        #     init_sigma_fn=lambda: 'test'
        # )
    )
    path = knn_detector.save(tmp_path)
    loaded_detector = load_detector(path)

    assert isinstance(loaded_detector, KNN)
    assert isinstance(loaded_detector.aggregator, TopKAggregator)
    assert isinstance(loaded_detector.normaliser, ShiftAndScaleNormaliser)
    # assert isinstance(loaded_detector.kernel, GaussianRBF)
    assert (loaded_detector.backend.ks.numpy() == np.array([8, 9, 10])).all()
    # assert loaded_detector.kernel.config['sigma'] == [1.0]
    assert loaded_detector.aggregator.k == 5
    assert loaded_detector.backend.__class__.__name__ == KNNTorch.__name__
    # assert loaded_detector.kernel.init_sigma_fn() == 'test'


def test_knn_to_torchscript(tmp_path):
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=TopKAggregator(k=2),
        normaliser=ShiftAndScaleNormaliser(),
        backend='pytorch'
    )

    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    knn_detector.infer_threshold(x_ref, 0.1)
    save_to_torch_script(knn_detector, tmp_path)


def test_ensemble_to_ts(tmp_path):
    tmp_path = './test'
    knn_detectors = [KNN(k=k+10) for k in range(2)]
    ensemble_detector = Ensemble(
        detectors=knn_detectors,
        aggregator=AverageAggregator(),
        normaliser=PValNormaliser(),
    )

    x_ref = torch.tensor(np.random.randn(100, 2))
    ensemble_detector.fit(x_ref)
    ensemble_detector.infer_threshold(x_ref, 0.1)
    ensemble_detector.save_modules(tmp_path)


def test_knn_state(tmp_path):
    tmp_path = './test'
    knn_detector = KNN(
        k=[8, 9, 10],
        aggregator=TopKAggregator(k=5),
        normaliser=ShiftAndScaleNormaliser(),
        backend='pytorch',
    )
    x_ref = np.random.randn(100, 2)
    knn_detector.fit(x_ref)
    knn_detector.infer_threshold(x_ref, 0.1)
    print(knn_detector.predict(np.array([[10, 1]])))

    path = knn_detector.save(tmp_path)
    knn_detector.save_state(tmp_path)
    loaded_detector = load_detector(path)
    loaded_detector.load_state(path)
    print(loaded_detector.predict(np.array([[10, 1]])))


def test_knn_and_transform_workflow(tmp_path):
    tmp_path = './test'
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)
    knn_1_save_path = tmp_path + '/knn_1'
    knn_2_save_path = tmp_path + '/knn_2'
    transform_save_path = tmp_path + '/transformer'

    knn_detector_1 = KNN(
        k=[8],
        backend='pytorch',
    )
    knn_detector_2 = KNN(
        k=[7],
        backend='pytorch',
    )

    x_ref = np.random.randn(100, 2)

    knn_detector_1.fit(x_ref)
    score_ref_1 = knn_detector_1.score(x_ref)
    save_to_torch_script(knn_detector_1, knn_1_save_path)
    # knn_detector_1.infer_threshold(x_ref, 0.1)
    # knn_detector_1.save(knn_1_save_path)
    # knn_detector_1.save_state(knn_1_save_path)

    knn_detector_2.fit(x_ref)
    x_ref_2 = knn_detector_2.score(x_ref)
    save_to_torch_script(knn_detector_2, knn_2_save_path)
    # knn_detector_2.infer_threshold(x_ref, 0.1)
    # knn_detector_2.save(knn_2_save_path)
    # knn_detector_2.save_state(knn_2_save_path)

    transform = Transform(transforms=[ShiftAndScaleNormaliser(), TopKAggregator(k=5)])
    x_ref = torch.concat((x_ref_1, x_ref_2), dim=1)

    transform.fit(x_ref)

    transform.save(transform_save_path)
    transform.save_state(transform_save_path)

    # loaded_knn_detector_1 = load_detector(knn_1_save_path)
    # loaded_knn_detector_1.load_state(knn_1_save_path)
    # assert loaded_knn_detector_1.backend.x_ref.shape == (100, 2)

    # loaded_knn_detector_2 = load_detector(knn_2_save_path)
    # loaded_knn_detector_2.load_state(knn_1_save_path)
    # assert loaded_knn_detector_2.backend.x_ref.shape == (100, 2)

    loaded_transform = load_detector(transform_save_path)
    assert loaded_transform.transforms[1].k == 5
    loaded_transform.load_state(transform_save_path)
    assert loaded_transform.transforms[0].fitted
    assert loaded_transform.transforms[0].val_means.shape == (1, 2)

