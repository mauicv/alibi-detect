import torch


def save_to_torch_script(detector, location):
    objs = [getattr(detector, attr) for attr in detector.MODULES
            if getattr(detector, attr) is not None]
    threshold_inferred = detector.threshold_inferred

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.objs = torch.nn.ModuleList(objs)
            self.threshold_inferred = threshold_inferred
            if self.threshold_inferred:
                self.val_scores = detector.val_scores
                self.threshold = detector.threshold

        def forward(self, X):
            for obj in self.objs:
                X = obj(X)
            if self.threshold_inferred:
                preds = X > self.threshold
                return preds
            else:
                return X

    detector = Detector()
    detector = torch.jit.script(detector)
    detector.save(location)
