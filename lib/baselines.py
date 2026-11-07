import numpy as np
import torch.nn as nn
from collections import Counter

# 17 joints * (mean_x, mean_y, std_x, std_y, mean_speed) = 85
POSE_STAT_DIM = 17 * 5


def pose_stat_features(x):
    """x: tensor (3, 90, 17) -> 1D feature vector of length POSE_STAT_DIM."""
    xy = x[:2].numpy()                      # (2, 90, 17)
    mean = xy.mean(axis=1)                  # (2, 17)
    std = xy.std(axis=1)                    # (2, 17)
    vel = np.diff(xy, axis=1)               # (2, 89, 17)
    speed = np.sqrt((vel ** 2).sum(axis=0)).mean(axis=0)  # (17,)
    return np.concatenate([mean.ravel(), std.ravel(), speed.ravel()]).astype(np.float32)


class MajorityBaseline:
    def __init__(self):
        self.mode_ = 0

    def fit(self, y):
        self.mode_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, n):
        return np.full(n, self.mode_, dtype=int)


class PoseStatsMLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class KeypointGRU(nn.Module):
    def __init__(self, num_classes, hidden=64):
        super().__init__()
        self.gru = nn.GRU(input_size=34, hidden_size=hidden,
                          num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: (B, 3, 90, 17) -> use xy only -> (B, 90, 34)
        b = x.size(0)
        xy = x[:, :2].permute(0, 2, 1, 3).reshape(b, 90, 34)
        _, h = self.gru(xy)
        return self.fc(h[-1])
