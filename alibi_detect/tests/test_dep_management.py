"""Test optional dependencies.
These tests import all the named objects from the public API of alibi-detect and test that they throw the correct errors
if the relevant optional dependencies are not installed. If these tests fail, it is likely that:
1. The optional dependency relation hasn't been added to the test script. In this case, this test assumes that the
functionality should work for the default alibi-detect install. If this is not the case the exported object name should
be added to the dependency_map in the relevant test.
2. The relevant export in the public API hasn't been imported using `optional_import` from
`alibi_detect.utils.missing_optional_dependency`.
Notes:
    1. These tests will be skipped in the normal test suite. To run correctly use tox.
    2. If you need to configure a new optional dependency you will need to update the setup.cfg file and add a testenv
    environment.
    3. Backend functionality may be unique to specific explainers/functions and so there may be multiple such modules
    that need to be tested separately.
"""

from types import ModuleType
from collections import defaultdict

import pytest


def check_correct_dependencies(
        module: ModuleType,
        dependencies: defaultdict,
        opt_dep: str):
    """Checks that imported modules that depend on optional dependencies throw correct errors on use.
    Parameters
    ----------
    module
        The module to check. Each of the public objects within this module will be checked.
    dependencies
        A dictionary mapping the name of the object to the list of optional dependencies that it depends on. If a name
        is not in the dictionary, the named object is assumed to be independent of optional dependencies. Therefor it
        should pass for the default alibi-detect install.
    opt_dep
        The name of the optional dependency that is being tested.
    """
    lib_obj = [obj for obj in dir(module) if not obj.startswith('_')]
    for item_name in lib_obj:
        item = getattr(module, item_name)
        if not isinstance(item, ModuleType):
            pass_contexts = dependencies[item_name]  # type: ignore
            if opt_dep in pass_contexts or 'default' in pass_contexts or opt_dep == 'all':
                with pytest.raises(AttributeError):
                    item.test  # type: ignore # noqa
            else:
                with pytest.raises(ImportError):
                    item.test  # type: ignore # noqa


def test_cd_dependencies(opt_dep):
    """Tests that the cd module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in []:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect import cd
        check_correct_dependencies(cd, dependency_map, opt_dep)


def test_cd_torch_dependencies(opt_dep):
    """Tests that the cd module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
        ("HiddenOutput", ['torch']),
        ("preprocess_drift", ['torch'])
    ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect.cd import pytorch as cd_pytorch
        check_correct_dependencies(cd_pytorch, dependency_map, opt_dep)


def test_cd_tensorflow_dependencies(opt_dep):
    """Tests that the cd module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
        ("HiddenOutput", ['tensorflow', 'tf-tfp']),
        ("UAE", ['tensorflow', 'tf-tfp']),
        ("preprocess_drift", ['tensorflow', 'tf-tfp'])
    ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect.cd import tensorflow as tensorflow_cd
        check_correct_dependencies(tensorflow_cd, dependency_map, opt_dep)


def test_ad_dependencies(opt_dep):
    """Tests that the ad module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ('AdversarialAE', ['tensorflow', 'tf-tfp']),
            ('ModelDistillation', ['tensorflow', 'tf-tfp'])
            ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect import ad
        check_correct_dependencies(ad, dependency_map, opt_dep)


def test_od_dependencies(opt_dep):
    """Tests that the od module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ('LLR', ['tf-tfp']),
            ('OutlierVAE', ['tf-tfp']),
            ('OutlierVAEGMM', ['tf-tfp']),
            ('OutlierAE', ['tensorflow', 'tf-tfp']),
            ('OutlierAEGMM', ['tensorflow', 'tf-tfp']),
            ('OutlierSeq2Seq', ['tensorflow', 'tf-tfp']),
            ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect import od
        check_correct_dependencies(od, dependency_map, opt_dep)


def test_tensorflow_model_dependencies(opt_dep):
    """Tests that the tensorflow models module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ("AE", ['tensorflow', 'tf-tfp']),
            ("AEGMM", ['tensorflow', 'tf-tfp']),
            ("Seq2Seq", ['tensorflow', 'tf-tfp']),
            ("VAE", ['tensorflow', 'tf-tfp']),
            ("VAEGMM", ['tensorflow', 'tf-tfp']),
            ("resnet", ['tensorflow', 'tf-tfp']),
            ("PixelCNN", ['tf-tfp']),
            ("TransformerEmbedding", ['tensorflow', 'tf-tfp']),
            ("trainer", ['tensorflow', 'tf-tfp']),
            ("eucl_cosim_features", ['tensorflow', 'tf-tfp']),
            ("elbo", ['tf-tfp']),
            ("loss_vaegmm", ['tf-tfp']),
            ("loss_aegmm", ['tensorflow', 'tf-tfp']),
            ("loss_adv_ae", ['tensorflow', 'tf-tfp']),
            ("loss_distillation", ['tensorflow', 'tf-tfp']),
            ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect.models import tensorflow as tf_models
        check_correct_dependencies(tf_models, dependency_map, opt_dep)


def test_torch_model_dependencies(opt_dep):
    """Tests that the torch models module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ("TransformerEmbedding", ['torch']),
            ("trainer", ['torch']),
            ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect.models import pytorch as torch_models
        check_correct_dependencies(torch_models, dependency_map, opt_dep)


def test_dataset_dependencies(opt_dep):
    """Tests that the datasets module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in []:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect import datasets
        check_correct_dependencies(datasets, dependency_map, opt_dep)


def test_fetching_utils_dependencies(opt_dep):
    """Tests that the fetching utils module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
        ('fetch_detector', ['tensorflow', 'tf-tfp']),
        ('fetch_tf_model', ['tensorflow', 'tf-tfp'])
    ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect.utils import fetching
        check_correct_dependencies(fetching, dependency_map, opt_dep)


def test_saving_utils_dependencies(opt_dep):
    """Tests that the saving utils module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
        ('load_detector', ['tf-tfp']),
        ('save_detector', ['tf-tfp'])
    ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect.utils import saving
        check_correct_dependencies(saving, dependency_map, opt_dep)


def test_tensorflow_utils_dependencies(opt_dep):
    """Tests that the saving utils module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
            ("batch_compute_kernel_matrix", ['tensorflow', 'tf-tfp']),
            ("mmd2", ['tensorflow', 'tf-tfp']),
            ("mmd2_from_kernel_matrix", ['tensorflow', 'tf-tfp']),
            ("relative_euclidean_distance", ['tensorflow', 'tf-tfp']),
            ("squared_pairwise_distance", ['tensorflow', 'tf-tfp']),
            ("GaussianRBF", ['tensorflow', 'tf-tfp']),
            ("DeepKernel", ['tensorflow', 'tf-tfp']),
            ("permed_lsdds", ['tensorflow', 'tf-tfp']),
            ("predict_batch", ['tensorflow', 'tf-tfp']),
            ("predict_batch_transformer", ['tensorflow', 'tf-tfp']),
            ("quantile", ['tensorflow', 'tf-tfp']),
            ("subset_matrix", ['tensorflow', 'tf-tfp']),
            ("zero_diag", ['tensorflow', 'tf-tfp'])
            ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect.utils import tensorflow as tensorflow_utils
        check_correct_dependencies(tensorflow_utils, dependency_map, opt_dep)


def test_pytorch_utils_dependencies(opt_dep):
    """Tests that the pytorch utils module correctly protects against uninstalled optional dependencies.
    """

    dependency_map = defaultdict(lambda: ['default'])
    for dependency, relations in [
        ("batch_compute_kernel_matrix", ['torch']),
        ("mmd2", ['torch']),
        ("mmd2_from_kernel_matrix", ['torch']),
        ("squared_pairwise_distance", ['torch']),
        ("GaussianRBF", ['torch']),
        ("DeepKernel", ['torch']),
        ("permed_lsdds", ['torch']),
        ("predict_batch", ['torch']),
        ("predict_batch_transformer", ['torch']),
        ("quantile", ['torch']),
        ("zero_diag", ['torch']),
        ("TorchDataset", ['torch']),
        ("get_device", ['torch']),
    ]:
        dependency_map[dependency] = relations
    if opt_dep != 'all':
        from alibi_detect.utils import pytorch as pytorch_utils
        check_correct_dependencies(pytorch_utils, dependency_map, opt_dep)


# def test_utils_dependencies(opt_dep):
#     """Tests that the datasets module correctly protects against uninstalled optional dependencies.
#     """
#
#     dependency_map = defaultdict(lambda: ['default'])
#     for dependency, relations in []:
#         dependency_map[dependency] = relations
#     if opt_dep != 'all':
#         from alibi_detect import utils
#         check_correct_dependencies(utils, dependency_map, opt_dep)
