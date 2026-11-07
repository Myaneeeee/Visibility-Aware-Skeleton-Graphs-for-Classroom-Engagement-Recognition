import torch
import baselines


def test_pose_stat_features_fixed_length():
    x = torch.rand(3, 90, 17)
    f = baselines.pose_stat_features(x)
    assert f.ndim == 1
    assert f.shape[0] == baselines.POSE_STAT_DIM


def test_majority_baseline_predicts_mode():
    m = baselines.MajorityBaseline()
    m.fit([0, 0, 0, 1])
    preds = m.predict(3)
    assert list(preds) == [0, 0, 0]


def test_mlp_forward_shape():
    net = baselines.PoseStatsMLP(in_dim=baselines.POSE_STAT_DIM, num_classes=2)
    out = net(torch.rand(4, baselines.POSE_STAT_DIM))
    assert out.shape == (4, 2)


def test_gru_forward_shape():
    net = baselines.KeypointGRU(num_classes=2)
    out = net(torch.rand(4, 3, 90, 17))
    assert out.shape == (4, 2)
