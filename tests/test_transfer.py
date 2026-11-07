import torch
import model as model_mod
from model import STGCNClassifier
from pretrain_ntu import export_backbone
import transfer


def _make_backbone_ckpt(tmp_path):
    src = STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=60)
    bb = export_backbone(src.state_dict())
    p = tmp_path / "bb.pth"
    torch.save(bb, p)
    return src, str(p)


def test_load_backbone_transfers_weights_and_reinits_fc(tmp_path):
    src, ckpt = _make_backbone_ckpt(tmp_path)
    model = STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=2)
    info = transfer.load_pretrained_backbone(model, ckpt)

    # backbone weights now equal the source
    assert torch.allclose(model.data_bn.weight, src.data_bn.weight)
    assert torch.allclose(model.layer1.gcn.conv.weight, src.layer1.gcn.conv.weight)
    # fc is the new 2-class head, not loaded from the 60-class source
    assert model.fc.out_features == 2
    assert any(k.startswith("fc.") for k in info["skipped"])
    assert all(not k.startswith("fc.") for k in info["loaded"])


def test_loaded_model_forwards_to_two_classes(tmp_path):
    _, ckpt = _make_backbone_ckpt(tmp_path)
    model = STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=2)
    transfer.load_pretrained_backbone(model, ckpt)
    out = model(torch.rand(4, 3, 90, 17))
    assert out.shape == (4, 2)


def test_backbone_transfers_onto_a_subset_graph(tmp_path):
    """E1 keeps config C's NTU warm start while changing only the topology.

    Every learned tensor in the backbone is node-agnostic -- the graph convs are
    1x1 and the temporal convs are (9, 1) -- so the only thing in the checkpoint
    with a node dimension is the adjacency buffer, which is topology rather than
    a weight. Without this, E1 could not warm-start and would silently be a
    different rung than the C it is compared against.
    """
    src, ckpt = _make_backbone_ckpt(tmp_path)
    model = STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=2,
                            joints=model_mod.OBSERVED_JOINTS_13)
    transfer.load_pretrained_backbone(model, ckpt)

    assert torch.allclose(model.layer1.gcn.conv.weight, src.layer1.gcn.conv.weight)
    assert model(torch.rand(4, 3, 90, 13)).shape == (4, 2)


def test_subset_model_keeps_its_own_adjacency(tmp_path):
    """The checkpoint's 17-node graph must not be loaded over the 13-node one."""
    _, ckpt = _make_backbone_ckpt(tmp_path)
    model = STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=2,
                            joints=model_mod.OBSERVED_JOINTS_13)
    transfer.load_pretrained_backbone(model, ckpt)
    expected = model_mod.build_spatial_partition_matrix(
        joints=model_mod.OBSERVED_JOINTS_13)
    assert torch.equal(model.layer1.gcn.A, expected)


def test_full_graph_model_still_ends_up_with_the_checkpoint_adjacency(tmp_path):
    """E0-reproduces-C: nothing about the 17-joint load path may move."""
    src, ckpt = _make_backbone_ckpt(tmp_path)
    model = STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=2)
    transfer.load_pretrained_backbone(model, ckpt)
    assert torch.equal(model.layer1.gcn.A, src.layer1.gcn.A)
