import os
from typing import Optional, List
from alibi_detect.od.base import OutlierDetector
from alibi_detect.od.aggregation import BaseTransform, AverageAggregator, PValNormaliser
from alibi_detect.od.processor import BaseProcessor
from alibi_detect.od.config import ConfigMixin
from alibi_detect.saving.registry import registry
from alibi_detect.od.loading import load_detector
from alibi_detect.od.backends.torch.saving import save_to_torch_script
import torch


@registry.register('Ensemble')
class Ensemble(OutlierDetector, ConfigMixin):
    CONFIG_PARAMS = ('detectors', 'aggregator', 'normaliser', 'processor')
    LARGE_PARAMS = ('detectors', )
    BASE_OBJ = True
    MODULES = ('detectors', 'normaliser', 'aggregator')

    def __init__(
            self,
            detectors: List[OutlierDetector],
            aggregator: Optional[BaseTransform] = AverageAggregator(),
            normaliser: Optional[BaseTransform] = PValNormaliser(),
            processor=BaseProcessor()):
        self._set_config(locals())

        self.detectors = detectors
        self.normaliser = normaliser
        self.aggregator = aggregator
        self.processor = processor

    def fit(self, X):
        for detector in self.detectors:
            detector.fit(X)

    def score(self, X):
        return torch.tensor(self.processor(X, self.detectors))

    @classmethod
    def _detectors_deserializer(self, key, val):
        return [load_detector(d) for d in val]

    def save_modules(self, path):
        if not os.path.isdir(str(path)):
            os.mkdir(str(path))
        os.mkdir(str(path)+'/detectors')
        os.mkdir(str(path)+'/aggregator')

        normaliser = self.normaliser
        aggregator = self.aggregator
        val_scores = self.val_scores
        threshold = self.threshold

        class Aggregator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.normaliser = normaliser
                self.aggregator = aggregator
                self.val_scores = val_scores
                self.threshold = threshold

            def forward(self, x):
                s = self.aggregator(self.normaliser(x))
                preds = s > self.threshold
                return preds

        aggregator = Aggregator()
        aggregator = torch.jit.script(aggregator)
        aggregator.save(str(path)+'/aggregator/model.pt')

        for ind, detector in enumerate(self.detectors):
            os.mkdir(str(path)+f'/detectors/{ind}')
            save_to_torch_script(detector, str(path)+f'/detectors/{ind}')
