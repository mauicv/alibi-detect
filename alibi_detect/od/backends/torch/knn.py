import numpy as np
import torch


class KNNTorch(torch.nn.Module):
    STATE = ('x_ref',)

    def __init__(self, k):
        super().__init__()
        self.ensemble = isinstance(k, (list, np.ndarray))
        ks = torch.tensor(k) if self.ensemble else torch.tensor([k])
        self.register_buffer('ks', ks, persistent=True)
        self.ensemble = isinstance(k, (np.ndarray, list, tuple))

    def score(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        K = torch.cdist(X, self.x_ref)
        bot_k_dists = torch.topk(K, self.ks.max(), dim=1, largest=False)
        all_knn_dists = bot_k_dists.values[:, self.ks-1]
        all_knn_dists = all_knn_dists if self.ensemble else all_knn_dists[:, 0]
        return all_knn_dists.cpu()

    def forward(self, X):
        return self.score(X)

    def fit(self, X):
        # self.x_ref = torch.as_tensor(X, dtype=torch.float32)
        self.register_buffer('x_ref', torch.as_tensor(X, dtype=torch.float32), persistent=True)
