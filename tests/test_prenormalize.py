import torch
import common


def test_prenormalize_preserves_shape_and_conf():
    kp = torch.rand(3, 90, 17) * 500 + 100
    kp[2] = torch.rand(90, 17) * 0.5 + 0.5   # all confident (> 0.01)
    conf_before = kp[2].clone()
    out = common.prenormalize_2d(kp)
    assert out.shape == (3, 90, 17)
    assert torch.allclose(out[2], conf_before)          # conf channel untouched


def test_prenormalize_centers_and_scales_to_unit_range():
    # x,y span a wide, known pixel extent; all joints confident
    kp = torch.zeros(3, 10, 17)
    kp[0] = torch.linspace(0, 640, 17).repeat(10, 1)    # x in [0, 640]
    kp[1] = torch.linspace(0, 480, 17).repeat(10, 1)    # y in [0, 480]
    kp[2] = torch.ones(10, 17)                          # confident
    out = common.prenormalize_2d(kp)
    assert abs(float(out[0].mean())) < 1e-3             # centered ~0
    assert abs(out[0].min() + 1.0) < 1e-3 and abs(out[0].max() - 1.0) < 1e-3
    assert abs(out[1].min() + 1.0) < 1e-3 and abs(out[1].max() - 1.0) < 1e-3


def test_prenormalize_zeros_low_confidence_joints():
    kp = torch.rand(3, 5, 17) * 500 + 100
    kp[2] = torch.ones(5, 17)
    kp[2, :, 3] = 0.0                                   # joint 3 not confident
    out = common.prenormalize_2d(kp)
    assert torch.all(out[0, :, 3] == 0.0)              # x zeroed
    assert torch.all(out[1, :, 3] == 0.0)              # y zeroed
    assert torch.any(out[0, :, 0] != 0.0)              # confident joint kept


def test_prenormalize_noop_when_extent_below_guard():
    # extent <= 10px in x -> coords left unchanged (no center/scale)
    kp = torch.zeros(3, 4, 17)
    kp[0] = 100.0 + torch.rand(4, 17) * 5               # x span < 10
    kp[1] = 100.0 + torch.rand(4, 17) * 5               # y span < 10
    kp[2] = torch.ones(4, 17)
    out = common.prenormalize_2d(kp)
    assert torch.allclose(out[0], kp[0]) and torch.allclose(out[1], kp[1])
