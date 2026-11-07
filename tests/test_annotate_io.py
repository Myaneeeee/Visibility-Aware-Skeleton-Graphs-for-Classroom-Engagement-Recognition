"""The annotator's disk + keyboard contract.

These pin the things that break a labeling session in progress rather than at
startup: a save that crashes after the first judgement, a header that silently
eats a label, a Skip that leaves no trace for adjudication.
"""
import annotate


def test_save_annotation_creates_missing_parent_dir(tmp_path):
    """The calibration round writes to datasets/sav/calibration/annotator_N.csv,
    a directory that does not exist yet. Without this the first label an
    annotator presses raises FileNotFoundError."""
    target = tmp_path / "calibration" / "annotator_1.csv"

    annotate.save_annotation("vidA_1_id_0_f0_f74", "Focused", str(target))

    assert target.read_text().splitlines() == [
        "window_id,label",
        "vidA_1_id_0_f0_f74,Focused",
    ]


def test_save_annotation_writes_header_into_an_empty_file(tmp_path):
    """An existing-but-empty CSV (interrupted run, `touch`, empty checkout) must
    still get a header -- every loader in the repo skips row 1, so a headerless
    file loses its first real label silently."""
    p = tmp_path / "annotations.csv"
    p.write_text("")

    annotate.save_annotation("vidA_1_id_0_f0_f74", "Focused", str(p))

    assert p.read_text().splitlines()[0] == "window_id,label"


def test_save_annotation_appends_without_repeating_header(tmp_path):
    p = tmp_path / "annotations.csv"
    annotate.save_annotation("w1", "Focused", str(p))
    annotate.save_annotation("w2", "Distracted", str(p))

    assert p.read_text().splitlines() == [
        "window_id,label", "w1,Focused", "w2,Distracted",
    ]


def test_load_existing_annotations_ignores_the_header_row(tmp_path):
    p = tmp_path / "annotations.csv"
    p.write_text("window_id,label\nw1,Focused\nw2,Skip\n")

    assert annotate.load_existing_annotations(str(p)) == {"w1", "w2"}


def test_load_existing_annotations_missing_file_is_empty(tmp_path):
    assert annotate.load_existing_annotations(str(tmp_path / "nope.csv")) == set()


def test_skipped_windows_are_recorded_and_not_reoffered(tmp_path):
    """Phase C adjudicates every Skip, so a Skip has to leave a row. It also
    means a resumed session does not re-offer what was already judged
    undecidable."""
    p = tmp_path / "annotations.csv"
    annotate.save_annotation("w1", annotate.SKIP_LABEL, str(p))

    assert annotate.load_existing_annotations(str(p)) == {"w1"}
    assert p.read_text().splitlines()[-1] == "w1,Skip"


def test_skip_label_is_not_a_trainable_class():
    """Skip must never reach the model or the agreement metric."""
    import common
    import annotation_report as ar
    assert annotate.SKIP_LABEL not in common.CLASS_TO_IDX
    assert annotate.SKIP_LABEL not in ar._VALID


class TestPreflight:
    """The data ships to two other machines by hand. A missing directory or a
    wrong relative path should be caught by one command, not discovered ten
    windows into a session."""

    def _windows(self, tmp_path, video_name="vidA_1", n_frames=54):
        import torch
        wdir = tmp_path / "windows_filtered"
        wdir.mkdir()
        w = {"video_name": video_name, "track_id": 7, "start_frame": 0,
             "end_frame": n_frames - 1, "n_native_frames": n_frames,
             "keypoints": torch.zeros(3, n_frames, 17),
             "bboxes": [[10.0, 10.0, 20.0, 20.0]] * n_frames}
        torch.save([w], wdir / f"{video_name}_windows.pt")
        return wdir, annotate_window_id(w)

    def test_passes_when_everything_is_present(self, tmp_path):
        wdir, wid = self._windows(tmp_path)
        vdir = tmp_path / "videos"
        vdir.mkdir()
        (vdir / "vidA_1.mp4").write_bytes(b"")
        listfile = tmp_path / "shard.txt"
        listfile.write_text(wid + "\n")

        ok, problems = annotate.preflight(
            str(wdir), str(vdir), str(tmp_path / "out" / "a.csv"), str(listfile))

        assert ok, problems

    def test_reports_a_missing_clip_video(self, tmp_path):
        wdir, wid = self._windows(tmp_path)
        vdir = tmp_path / "videos"
        vdir.mkdir()                                   # exists but empty
        listfile = tmp_path / "shard.txt"
        listfile.write_text(wid + "\n")

        ok, problems = annotate.preflight(
            str(wdir), str(vdir), str(tmp_path / "a.csv"), str(listfile))

        assert not ok
        assert any("vidA_1.mp4" in p for p in problems)

    def test_reports_list_ids_with_no_window_on_disk(self, tmp_path):
        wdir, _ = self._windows(tmp_path)
        vdir = tmp_path / "videos"
        vdir.mkdir()
        (vdir / "vidA_1.mp4").write_bytes(b"")
        listfile = tmp_path / "shard.txt"
        listfile.write_text("vidZ_9_id_0_f0_f53\n")     # not in windows_filtered

        ok, problems = annotate.preflight(
            str(wdir), str(vdir), str(tmp_path / "a.csv"), str(listfile))

        assert not ok
        assert any("vidZ_9_id_0_f0_f53" in p for p in problems)

    def test_reports_an_empty_windows_dir(self, tmp_path):
        wdir = tmp_path / "windows_filtered"
        wdir.mkdir()
        vdir = tmp_path / "videos"
        vdir.mkdir()

        ok, problems = annotate.preflight(str(wdir), str(vdir), str(tmp_path / "a.csv"), None)

        assert not ok
        assert any("no *_windows.pt" in p for p in problems)


def annotate_window_id(w):
    import common
    return common.make_window_id(w)


class TestResolveKey:
    """Key handling, without a GUI. The on-screen legend promises 'D' for skip;
    a shifted or caps-locked 'D' must do the same thing as 'd'."""

    def test_class_keys(self):
        assert annotate.resolve_key(ord("1")) == "Focused"
        assert annotate.resolve_key(ord("2")) == "Distracted"
        assert annotate.resolve_key(ord("8")) == "TRASH"
        assert annotate.resolve_key(ord("9")) == "QUIT"

    def test_skip_is_case_insensitive(self):
        assert annotate.resolve_key(ord("d")) == "skip"
        assert annotate.resolve_key(ord("D")) == "skip"

    def test_space_toggles_skeleton(self):
        assert annotate.resolve_key(32) == "toggle"

    def test_no_keypress_and_unbound_keys_do_nothing(self):
        assert annotate.resolve_key(255) is None      # cv2 "no key" on some builds
        assert annotate.resolve_key(-1) is None       # cv2 "no key"
        assert annotate.resolve_key(ord("q")) is None
        assert annotate.resolve_key(ord("5")) is None
