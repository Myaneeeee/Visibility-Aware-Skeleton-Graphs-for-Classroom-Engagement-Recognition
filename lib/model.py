"""ST-GCN model: COCO-17 graph topology + spatial-temporal graph-conv classifier.

(Was the model half of the old p2_model.py; the windowing half moved to
steps/make_windows.py. This file is git-moved to lib/model.py.)
"""
import numpy as np
import torch
import torch.nn as nn


# ==========================================
# 1. GRAPH TOPOLOGY (ST-GCN 3-PARTITION)
# ==========================================
NUM_COCO_JOINTS = 17

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),         # Face
    (0, 5), (0, 6),                         # Neck/Shoulder connection
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),# Upper Body / Arms
    (5, 11), (6, 12), (11, 12),             # Torso / Pelvis
    (11, 13), (13, 15), (12, 14), (14, 16)  # Lower Body / Legs
]

# The joints whose mean detector confidence clears the 0.35 usability threshold
# in classroom footage: knees (0.29) and ankles (0.19) do not, because students
# sit behind desks. Measured over every frame of every labeled window by
# steps/diagnose_representation.py -> outputs/diagnostics/representation.json.
OBSERVED_JOINTS_13 = list(range(13))

# The same measurement as a continuous weight, for rung E2. Mean detector
# confidence per COCO joint over every frame of every labeled window, measured
# once and fixed before E2 ran.
COCO_JOINT_CONFIDENCE = [
    0.9258720278739929,     #  0 nose
    0.8255227208137512,     #  1 eye_L
    0.8231580257415771,     #  2 eye_R
    0.5764970779418945,     #  3 ear_L
    0.5670322179794312,     #  4 ear_R
    0.9604452252388,        #  5 shoulder_L
    0.9609599113464355,     #  6 shoulder_R
    0.7877274751663208,     #  7 elbow_L
    0.8044260740280151,     #  8 elbow_R
    0.7121819853782654,     #  9 wrist_L
    0.730613112449646,      # 10 wrist_R
    0.7046403288841248,     # 11 hip_L
    0.7163149118423462,     # 12 hip_R
    0.2908518612384796,     # 13 knee_L
    0.296784371137619,      # 14 knee_R
    0.1905089020729065,     # 15 ankle_L
    0.19079901278018951,    # 16 ankle_R
]


def _joint_positions(joints):
    """Map original COCO joint id -> its index in the subset. Validates the subset."""
    if len(set(joints)) != len(joints):
        raise ValueError(f"duplicate joints in {joints}")
    if any(j < 0 or j >= NUM_COCO_JOINTS for j in joints):
        raise ValueError(f"joints must lie in [0, {NUM_COCO_JOINTS}): {joints}")
    return {j: pos for pos, j in enumerate(joints)}


def get_coco_edges(joints=None):
    """Skeleton edges, reindexed onto `joints` if a subset is given.

    Edges with an endpoint outside the subset are dropped, so the remaining
    joints keep only the connections that survive.
    """
    if joints is None:
        return list(COCO_EDGES)
    pos = _joint_positions(joints)
    return [(pos[i], pos[j]) for i, j in COCO_EDGES if i in pos and j in pos]


def bone_parents(joints=None, center_node=0):
    """Parent of each node in the skeleton tree rooted at `center_node`.

    A bone vector is a joint minus its parent, so the root is its own parent and
    carries the zero vector. A joint the subset has severed from the root roots
    itself rather than pointing outside the graph.
    """
    joints = list(range(NUM_COCO_JOINTS)) if joints is None else list(joints)
    pos = _joint_positions(joints)
    if center_node not in pos:
        raise ValueError(
            f"center_node {center_node} is not in the joint subset {list(joints)}")
    root = pos[center_node]

    neighbours = {i: set() for i in range(len(joints))}
    for i, j in get_coco_edges(joints=joints):
        neighbours[i].add(j)
        neighbours[j].add(i)

    parents = [None] * len(joints)
    parents[root] = root
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nb in sorted(neighbours[node]):
            if parents[nb] is None:
                parents[nb] = node
                queue.append(nb)
    return [root if p is None else p for p in parents]


def _raw_partitions(joints, center_node):
    """Unnormalized 0/1 partition tensor (3, V, V): self, centripetal, centrifugal."""
    pos = _joint_positions(joints)
    if center_node not in pos:
        raise ValueError(
            f"center_node {center_node} is not in the joint subset {list(joints)}")
    num_nodes = len(joints)
    center = pos[center_node]
    edges = get_coco_edges(joints=joints)

    dist = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist, 0)
    for i, j in edges:
        dist[i, j] = 1
        dist[j, i] = 1

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    A = np.zeros((3, num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if dist[i, j] == 1 or i == j:
                if i == j:
                    A[0, i, j] = 1
                elif dist[j, center] < dist[i, center]:
                    A[1, i, j] = 1
                else:
                    A[2, i, j] = 1
    return A


def _normalize_partitions(A):
    """Row-normalize each partition, leaving all-zero rows alone."""
    A = np.array(A, dtype=np.float32, copy=True)
    for p in range(A.shape[0]):
        row_sum = np.sum(A[p], axis=1)
        row_sum[row_sum == 0] = 1
        D_inv = np.diag(1.0 / row_sum)
        A[p] = np.dot(D_inv, A[p])
    return A


def build_spatial_partition_matrix(joints=None, center_node=0, weights=None):
    """ST-GCN 3-partition adjacency over `joints` (default: all of COCO-17).

    `center_node` is an original COCO joint id and must be in the subset: the
    centripetal/centrifugal partitions are defined by distance to it.

    `weights` is a per-COCO-joint observability weight (rung E2). Each edge is
    scaled by the product of its endpoints' weights before normalization. Note
    that a row's own factor divides out under normalization, so this weights
    each incoming message by the confidence of the joint it comes from.
    """
    joints = list(range(NUM_COCO_JOINTS)) if joints is None else list(joints)
    A = _raw_partitions(joints, center_node)
    if weights is not None:
        if len(weights) != NUM_COCO_JOINTS:
            raise ValueError(
                f"weights must give one value per COCO joint "
                f"({NUM_COCO_JOINTS}), got {len(weights)}")
        w = np.asarray([weights[j] for j in joints], dtype=np.float32)
        A = A * w[None, :, None] * w[None, None, :]
    return torch.tensor(_normalize_partitions(A), dtype=torch.float32)

# ==========================================
# 2. PYTORCH ST-GCN MODEL BLOCKS
# ==========================================
class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix):
        super(SpatialGraphConv, self).__init__()
        self.register_buffer('A', adjacency_matrix)
        self.num_partitions = adjacency_matrix.shape[0]
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_partitions, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        n, kc, t, v = x.size()
        c = kc // self.num_partitions
        x = x.view(n, self.num_partitions, c, t, v)
        x = torch.einsum('npctw, pwv -> nctv', x, self.A)
        return x.contiguous()

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency_matrix, stride=1):
        super(STGCNBlock, self).__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, adjacency_matrix)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3)
        )

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)

class STGCNClassifier(nn.Module):
    """
    Standalone Spatio-Temporal Graph Convolutional Network (ST-GCN)
    configured for student action recognition.

    `joints` restricts the graph to a subset of COCO-17 (rung E1); inputs must
    then carry that many nodes. `weights` scales its edges by per-joint
    observability (rung E2). `readout` restricts the global average pool to
    those node indices (rung E5); None pools over every node. `observed` blocks
    every other node at the input (rung E6): data_bn's statistics are taken over
    those nodes alone and the rest leave it as zeros. `joint_bn` gives every
    (node, channel) pair its own data_bn statistics, as the reference ST-GCN
    does (rung E7); the default pools one set per channel over all nodes.
    """
    def __init__(self, in_channels=3, hidden_dim=128, num_classes=2, joints=None,
                 weights=None, readout=None, observed=None, joint_bn=False):
        super(STGCNClassifier, self).__init__()
        if joint_bn and observed is not None:
            raise ValueError("joint_bn and observed are separate rungs")
        self.readout = list(readout) if readout is not None else None
        self.observed = list(observed) if observed is not None else None
        self.joint_bn = joint_bn
        A = build_spatial_partition_matrix(joints=joints, weights=weights)

        # Initial normalization of skeletal coordinates
        if joint_bn:
            self.data_bn = nn.BatchNorm1d(in_channels * A.shape[-1])
        else:
            self.data_bn = nn.BatchNorm2d(in_channels)

        # Spatial-Temporal feature extraction backbone
        self.layer1 = STGCNBlock(in_channels, 32, A, stride=1)
        self.layer2 = STGCNBlock(32, 64, A, stride=2)
        self.layer3 = STGCNBlock(64, hidden_dim, A, stride=2)

        # Global average pooling collapses the temporal (T) and node (V) domains
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classification layer targeting Focused vs Distracted
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Input tensor shape expectations: (Batch, Channels, Time, Nodes)
        if self.joint_bn:
            # (N, C, T, V) -> (N, V*C, T), the reference layout: feature v*C + c.
            n, c, t, v = x.shape
            x = x.permute(0, 3, 1, 2).reshape(n, v * c, t)
            x = self.data_bn(x)
            x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        elif self.observed is None:
            x = self.data_bn(x)
        else:
            # data_bn pools over B, T and V, so an unobserved joint shifts the
            # statistics every later layer sees. Normalizing the observed slice
            # alone keeps it out of them; the rest stay at the post-norm mean.
            normalized = self.data_bn(x[:, :, :, self.observed])
            x = torch.zeros_like(x)
            x[:, :, :, self.observed] = normalized
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Pool space and time context down to 1D feature vector
        if self.readout is None:
            x = self.pool(x).view(x.size(0), -1)
        else:
            x = x[:, :, :, self.readout].mean(dim=(2, 3))

        # Logits generation
        out = self.fc(x)
        return out
