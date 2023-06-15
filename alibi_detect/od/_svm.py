from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np

from alibi_detect.base import (BaseDetector, FitMixin, ThresholdMixin,
                               outlier_prediction_dict)
from alibi_detect.exceptions import _catch_error as catch_error
from alibi_detect.od.pytorch import SVMTorch
from alibi_detect.od.sklearn import SVMSklearn
from alibi_detect.utils._types import Literal
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.version import __version__

if TYPE_CHECKING:
    import torch


backends = {
    'pytorch': SVMTorch,
    'sklearn': SVMSklearn
}


class SVM(BaseDetector, ThresholdMixin, FitMixin):
    def __init__(
        self,
        n_components: Optional[int] = None,
        backend: Literal['pytorch', 'sklearn'] = 'sklearn',
        device: Optional[Union[Literal['cuda', 'gpu', 'cpu'], 'torch.device']] = None,
        kernel: Union['torch.nn.Module', Literal['linear', 'poly', 'rbf', 'sigmoid']] = 'rbf',
        sigma: Optional[float] = None,
        kernel_params: Optional[Dict] = None,
    ) -> None:
        """Support vector machine (SVM) outlier detector.

        TODO

        # The Support vector machine outlier detector fits a one-class SVM to the reference data and uses the decision
        # function to detect outliers.

        # Rather than the typical approach of optimizing the exact kernel OCSVM objective through a dual formulation,
        # here we instead map the data into the kernel's RKHS and then solve the linear optimization problem
        # directly through its primal formulation. The Nystroem approximation can optionally be used to speed up
        # training and inference by approximating the kernel's RKHS.

        # This is the approach of SKLearn's `SGDOneClassSVM`, except here the optimization procedure is more tailored
        # for operation on GPUs. Instead of applying stochastic gradient descent (one data point at a time) with a
        # fixed learning rate schedule we perform full gradient descent with step size chosen at each iteration via
        # line search. Note that on a CPU this would not necessarily be preferable to SGD as we would have to iterate
        # through both data points and candidate step sizes, however on GPU all of the operations are
        # vectorized/parallelized.

        # Moreover, the Nystroem approximation has complexity `O(n^2m)` where `n` is the number of reference instances
        # and `m` defines the number of inducing points. This can therefore be expensive for large reference sets and
        # benefits from implementation on the GPU.

        # Parameters
        # ----------
        # kernel
        #     Used to define similarity between data points.
        # n_components
        #     Number of components in the Nystroem approximation By default uses all of them.
        # device
        #     Device type used. The default tries to use the GPU and falls back on CPU if needed. Can be specified by
        #     passing either ``'cuda'``, ``'gpu'``, ``'cpu'`` or an instance of ``torch.device``.
        # sigma
        #     Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        # kernel_params
        #     Additional parameters (keyword arguments) for kernel function passed as a dictionary.... TODO

        Raises
        ------
        NotImplementedError
            If choice of `backend` is not implemented.
        """
        super().__init__()

        backend_str: str = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch'], 'sklearn': ['sklearn']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend_str)

        backend_cls = backends[backend]
        args: Dict[str, Any] = {
            'n_components': n_components,
            'kernel': kernel,
            'sigma': sigma,
        }
        if backend == 'pytorch':
            args['device'] = device
        if backend == 'sklearn':
            args['kernel_params'] = kernel_params
        self.backend = backend_cls(**args)

    def fit(
        self,
        x_ref: np.ndarray,
        nu: float,
        step_size_range: Tuple[float, float] = (1e-6, 1.0),
        n_step_sizes: int = 16,
        tol: float = 1e-6,
        n_iter_no_change: int = 25,
        max_iter: int = 1000,
        verbose: int = 0,
    ) -> None:
        """Fit the detector on reference data.

        TODO

        Parameters
        ----------
        x_ref
            Training data.
        nu
            The proportion of the training data that should be considered outliers. Note that this does
            not necessarily correspond to the false positive rate on test data, which is still defined when
            calling the `infer_threshold()` method.
        step_size_range
            The range of values to be considered for the gradient descent step size at each iteration.
        n_step_sizes
            The number of step sizes in the defined range to be tested for loss reduction. This many points
            are spaced equidistantly along the range in log space.
        tol
            The decrease in loss required over the previous n_iter_no_change iterations in order to
            continue optimizing.
        n_iter_no_change
            The number of iterations over which the loss must decrease by `tol` in order for
            optimization to continue.
        max_iter
            The maximum number of optimization steps.
        verbose
            Verbosity level during training. 0 is silent, 1 a progress bar.
        """
        self.backend.fit(
            self.backend._to_tensor(x_ref),
            **self.backend.format_fit_kwargs(locals())
        )

    @catch_error('NotFittedError')
    def score(self, x: np.ndarray) -> np.ndarray:
        """Score `x` instances using the detector.

        TODO

        Parameters
        ----------
        x
            Data to score. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Outlier scores. The shape of the scores is `(n_instances,)`. The higher the score, the more anomalous the \
        instance.

        Raises
        ------
        NotFittedError
            If called before detector has been fit.
        """
        score = self.backend.score(self.backend._to_tensor(x))
        return self.backend._to_numpy(score)

    @catch_error('NotFittedError')
    def infer_threshold(self, x: np.ndarray, fpr: float) -> None:
        """Infer the threshold for the SVM detector.

        The threshold is computed so that the outlier detector would incorrectly classify `fpr` proportion of the
        reference data as outliers.

        Parameters
        ----------
        x
            Reference data used to infer the threshold.
        fpr
            False positive rate used to infer the threshold. The false positive rate is the proportion of
            instances in `x` that are incorrectly classified as outliers. The false positive rate should
            be in the range ``(0, 1)``.

        Raises
        ------
        ValueError
            Raised if `fpr` is not in ``(0, 1)``.
        NotFittedError
            If called before detector has been fit.
        """
        self.backend.infer_threshold(self.backend._to_tensor(x), fpr)

    @catch_error('NotFittedError')
    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """Predict whether the instances in `x` are outliers or not.

        Scores the instances in `x` and if the threshold was inferred, returns the outlier labels and p-values as well.

        Parameters
        ----------
        x
            Data to predict. The shape of `x` should be `(n_instances, n_features)`.

        Returns
        -------
        Dictionary with keys 'data' and 'meta'. 'data' contains the outlier scores. If threshold inference was  \
        performed, 'data' also contains the threshold value, outlier labels and p-vals . The shape of the scores is \
        `(n_instances,)`. The higher the score, the more anomalous the instance. 'meta' contains information about \
        the detector.

        Raises
        ------
        NotFittedError
            If called before detector has been fit.
        """
        outputs = self.backend.predict(self.backend._to_tensor(x))
        output = outlier_prediction_dict()
        output['data'] = {
            **output['data'],
            **self.backend._to_numpy(outputs)
        }
        output['meta'] = {
            **output['meta'],
            'name': self.__class__.__name__,
            'detector_type': 'outlier',
            'online': False,
            'version': __version__,
        }
        return output
