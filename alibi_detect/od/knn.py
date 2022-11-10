from typing import Literal, Union, Optional
import numpy as np

from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.aggregation import BaseTransform

from alibi_detect.od.backends import KNNTorch, KNNKeops
from alibi_detect.utils.frameworks import BackendValidator
from alibi_detect.od.config import ConfigMixin, StatefulMixin
from alibi_detect.saving.registry import registry


X_REF_FILENAME = 'x_ref.npy'

backends = {
    'pytorch': KNNTorch,
    'keops': KNNKeops
}


@registry.register('KNN')
class KNN(OutlierDetector, ConfigMixin, StatefulMixin):
    # ConfigMixin Parameters
    CONFIG_PARAMS = ('aggregator', 'normaliser', 'backend', 'k')
    BASE_OBJ = True
    LARGE_PARAMS = ()
    MODULES = ('backend', 'normaliser', 'aggregator')

    # StateMixin Parameters
    STATE = ('backend',)

    def __init__(
        self,
        k: Union[int, list],
        aggregator: Optional[BaseTransform] = None,
        normaliser: Optional[BaseTransform] = None,
        backend: Literal = 'pytorch'
    ) -> None:
        OutlierDetector.__init__(self)
        ConfigMixin.__init__(self)

        self._set_config(locals())
        backend = backend.lower()
        BackendValidator(
            backend_options={'pytorch': ['pytorch'],
                             'keops': ['keops']},
            construct_name=self.__class__.__name__
        ).verify_backend(backend)

        self.ensemble = isinstance(k, np.ndarray)
        self.normaliser = normaliser
        self.aggregator = aggregator
        self.fitted = False
        self.backend = backends[backend](k)

    def fit(self, X) -> None:
        self.backend.fit(X)
        val_scores = self.score(X)
        if getattr(self, 'normaliser'):
            self.normaliser.fit(val_scores)

    def score(self, X):
        y = self.backend.score(X)
        return y
