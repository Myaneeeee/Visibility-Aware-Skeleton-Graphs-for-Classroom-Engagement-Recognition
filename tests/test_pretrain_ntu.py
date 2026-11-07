import os
import pytest
import torch
from torch.utils.data import Dataset
import pretrain_ntu as pt


def test_export_backbone_strips_fc():
    from model import STGCNClassifier
    sd = STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=60).state_dict()
    bb = pt.export_backbone(sd)
    assert not any(k.startswith("fc.") for k in bb)
    assert any(k.startswith("data_bn.") for k in bb)
    assert any(k.startswith("layer1.") for k in bb)


class _TinyDS(Dataset):
    def __init__(self, n=8, num_classes=60):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(n, 3, 90, 17, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


def test_train_backbone_smoke_writes_backbone(tmp_path):
    out = tmp_path / "bb.pth"
    res = pt.train_backbone(_TinyDS(), _TinyDS(), epochs=1, batch_size=4,
                            device="cpu", out_path=str(out), patience=1)
    assert os.path.exists(out)
    sd = torch.load(out, map_location="cpu", weights_only=True)
    assert not any(k.startswith("fc.") for k in sd)      # exported backbone only
    assert "ckpt_path" in res and res["ckpt_path"] == str(out)


def test_train_backbone_rejects_empty_val(tmp_path):
    with pytest.raises(ValueError):
        pt.train_backbone(_TinyDS(), _TinyDS(n=0), epochs=1, batch_size=4,
                          device="cpu", out_path=str(tmp_path / "x.pth"), patience=1)
