import os
import extract_poses

def test_resolve_dirs_defaults_to_local_paths():
    inp, out = extract_poses.resolve_dirs(None, None)
    assert inp.endswith(os.path.join("datasets", "sav", "videos"))
    assert out.endswith(os.path.join("datasets", "sav", "pose_tracks"))

def test_resolve_dirs_uses_overrides():
    inp, out = extract_poses.resolve_dirs("/tmp/in", "/tmp/out")
    assert inp == "/tmp/in"
    assert out == "/tmp/out"

def test_order_videos_follows_the_order_file(tmp_path):
    files = [f"/v/{c}.mp4" for c in ("a", "b", "c")]
    order = tmp_path / "order.txt"
    order.write_text("c\na\nb\n")
    assert extract_poses.order_videos(files, str(order)) == ["/v/c.mp4", "/v/a.mp4", "/v/b.mp4"]

def test_order_videos_rejects_a_mismatched_order_file(tmp_path):
    import pytest
    order = tmp_path / "order.txt"
    order.write_text("a\nz\n")
    with pytest.raises(ValueError):
        extract_poses.order_videos(["/v/a.mp4", "/v/b.mp4"], str(order))
