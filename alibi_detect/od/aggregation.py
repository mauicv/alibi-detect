from __future__ import annotations
from typing import List
import logging

from alibi_detect.od.config import ConfigMixin, StatefulMixin
from alibi_detect.saving.registry import registry
import torch


logger = logging.getLogger(__name__)


class BaseTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fitted = False
        self.config = {}

    def fit(self, X):
        if not self.fitted and hasattr(self, '_fit'):
            self._fit(X)
            self.fitted = True
        return self

    def _fit(self, X):
        pass

    def transform(self, scores):
        if not self.fitted:
            raise Exception('Transform not fitted, call fit before calling transform!')
        return self._transform(scores)

    def _transform(self, scores):
        raise NotImplementedError

    def forward(self, scores):
        return self._transform(scores)


@registry.register('PValNormaliser')
class PValNormaliser(BaseTransform, ConfigMixin, StatefulMixin):
    STATE = ('val_scores', 'fitted')

    def __init__(self):
        ConfigMixin.__init__(self)
        BaseTransform.__init__(self)
        StatefulMixin.__init__(self)
        self._set_config(locals())

    def _fit(self, val_scores):
        self.val_scores = val_scores

    def _transform(self, scores):
        p_vals = (
                1 + (scores[:, None, :] < self.val_scores[None, :, :]).sum(1)
            )/(len(self.val_scores)+1)
        return 1 - p_vals


@registry.register('ShiftAndScaleNormaliser')
class ShiftAndScaleNormaliser(BaseTransform, ConfigMixin, StatefulMixin):
    STATE = ('val_means', 'val_scales', 'fitted')

    def __init__(self):
        ConfigMixin.__init__(self)
        BaseTransform.__init__(self)
        StatefulMixin.__init__(self)
        self._set_config(locals())

    def _fit(self, val_scores):
        self.val_means = val_scores.mean(0)[None, :]
        self.val_scales = val_scores.std(0)[None, :]

    def _transform(self, scores):
        return (scores - self.val_means)/self.val_scales


@registry.register('TopKAggregator')
class TopKAggregator(BaseTransform, ConfigMixin):
    CONFIG_PARAMS = ('k', )
    BASE_OBJ = False
    LARGE_PARAMS = ()

    def __init__(self, k=None):
        ConfigMixin.__init__(self)
        BaseTransform.__init__(self)
        self._set_config(locals())
        self.k = k
        self.fitted = True

    def _transform(self, scores):
        if self.k is None:
            self.k = int(torch.ceil(scores.shape[1]/2))
        return torch.sort(scores, 1)[0][:, -self.k:].mean(-1)


@registry.register('AverageAggregator')
class AverageAggregator(BaseTransform, ConfigMixin):
    CONFIG_PARAMS = ('weights', )
    BASE_OBJ = False
    LARGE_PARAMS = ()

    def __init__(self, weights=None):
        ConfigMixin.__init__(self)
        BaseTransform.__init__(self)
        self._set_config(locals())
        self.weights = weights
        self.fitted = True

    def _transform(self, scores):
        if self.weights is None:
            m = scores.shape[-1]
            self.weights = torch.ones(m)/m
        return scores @ self.weights


@registry.register('MaxAggregator')
class MaxAggregator(BaseTransform, ConfigMixin):
    CONFIG_PARAMS = ()

    def __init__(self, ConfigMixin):
        ConfigMixin.__init__(self)
        BaseTransform.__init__(self)
        self._set_config(locals())
        self.fitted = True

    def _transform(self, scores):
        return torch.max(scores, axis=-1)


@registry.register('MinAggregator')
class MinAggregator(BaseTransform, ConfigMixin):
    CONFIG_PARAMS = ()

    def __init__(self):
        ConfigMixin.__init__(self)
        BaseTransform.__init__(self)
        self._set_config(locals())
        self.fitted = True

    def _transform(self, scores):
        return torch.min(scores, axis=-1)


@registry.register('Transform')
class Transform(BaseTransform, ConfigMixin, StatefulMixin):
    # ConfigMixin Parameters
    BASE_OBJ = True
    CONFIG_PARAMS = ('transforms', )

    # StateMixin Parameters
    STATE = ('transforms', )

    def __init__(self, transforms):
        ConfigMixin.__init__(self)
        BaseTransform.__init__(self)
        self._set_config(locals())
        self.transforms: List[BaseTransform] = transforms
        self.fitted = False

    def _transform(self, X):
        for transform in self.transforms:
            X = transform(X)
        return X

    def _fit(self, X):
        for transfrom in self.transforms:
            transfrom.fit(X)
        self.fitted = True

    @classmethod
    def _transforms_deserializer(cls, key, configs):
        return [cls.deserialize_value('', cfg) for cfg in configs]
