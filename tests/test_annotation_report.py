import annotation_report as ar


def test_load_labels_skips_header_and_keeps_valid(tmp_path):
    p = tmp_path / "annotations.csv"
    p.write_text("window_id,label\nvidA_1_id_0_f0_f89,Focused\nvidA_1_id_1_f0_f89,Distracted\n")
    assert ar.load_labels(str(p)) == {
        "vidA_1_id_0_f0_f89": "Focused",
        "vidA_1_id_1_f0_f89": "Distracted",
    }


def test_summarize_counts_and_video_coverage():
    labels = {
        "vidA_1_id_0_f0_f89": "Focused",
        "vidA_1_id_1_f0_f89": "Distracted",
        "vidB_2_id_0_f0_f89": "Focused",
    }
    s = ar.summarize(labels)
    assert s["counts"]["Focused"] == 2
    assert s["counts"]["Distracted"] == 1
    assert s["total"] == 3
    assert s["source_videos"] == 2         # vidA, vidB


def test_prior_hit_rate_higher_in_distracted_prior_clips():
    clip_actions = {"vidA_9": {"p0": {"look_sideways"}}, "vidB_1": {"p0": {"read"}}}
    labels = {
        "vidA_9_id_0_f0_f89": "Distracted",
        "vidA_9_id_1_f0_f89": "Focused",
        "vidB_1_id_0_f0_f89": "Focused",
    }
    r = ar.prior_hit_rate(labels, clip_actions)
    assert r["distracted_rate_in_prior"] == 0.5     # 1 of 2
    assert r["distracted_rate_in_rest"] == 0.0      # 0 of 1


def test_cohen_kappa_perfect_and_none():
    a = {"w1": "Focused", "w2": "Distracted", "w3": "Focused"}
    assert ar.cohen_kappa(a, a) == 1.0
    b = {"w1": "Distracted", "w2": "Focused", "w3": "Distracted"}
    assert ar.cohen_kappa(a, b) < 0.5               # strong disagreement


def test_skips_are_listed_but_excluded_from_labels(tmp_path):
    """Phase C adjudicates every Skip, so the report has to surface them -- while
    keeping them out of class balance and out of kappa."""
    p = tmp_path / "annotations.csv"
    p.write_text("window_id,label\n"
                 "vidA_1_id_0_f0_f74,Focused\n"
                 "vidA_1_id_1_f0_f74,Skip\n"
                 "vidB_2_id_0_f0_f74,Skip\n")

    assert ar.load_labels(str(p)) == {"vidA_1_id_0_f0_f74": "Focused"}
    assert ar.load_skipped(str(p)) == ["vidA_1_id_1_f0_f74", "vidB_2_id_0_f0_f74"]


def test_kappa_ignores_windows_one_annotator_skipped():
    """A Skip is 'I cannot tell', not a category. Scoring it as a third class
    would drag kappa toward zero and misreport genuine agreement."""
    a = {"w1": "Focused", "w2": "Distracted", "w3": "Focused"}
    b = {"w1": "Focused", "w2": "Distracted"}          # w3 was skipped -> absent
    assert ar.cohen_kappa(a, b) == 1.0


def test_main_reports_skip_count(tmp_path, capsys):
    p = tmp_path / "annotations.csv"
    p.write_text("window_id,label\nvidA_1_id_0_f0_f74,Focused\nvidA_1_id_1_f0_f74,Skip\n")

    ar.main(annotations_csv=str(p))

    out = capsys.readouterr().out
    assert "1 skipped" in out
    assert "adjudication" in out


def test_main_prints_kappa_when_annotations_b_given(tmp_path, capsys):
    a = tmp_path / "a.csv"
    a.write_text("window_id,label\nvidA_1_id_0_f0_f89,Focused\nvidA_1_id_1_f0_f89,Distracted\n")
    b = tmp_path / "b.csv"
    b.write_text("window_id,label\nvidA_1_id_0_f0_f89,Focused\nvidA_1_id_1_f0_f89,Distracted\n")

    ar.main(annotations_csv=str(a), annotations_b=str(b))

    out = capsys.readouterr().out
    assert "kappa" in out


def _wid(video, clip, track):
    return f"{video}_{clip}_id_{track}_f0_f74"


def test_cv_group_report_groups_by_source_video_not_clip():
    """Two clips of one video are one CV group -- the unit evaluate.py splits on."""
    labels = {_wid("20230118_34", 1, 0): "Focused",
              _wid("20230118_34", 9, 0): "Distracted",
              _wid("20220716_44", 3, 0): "Focused"}
    r = ar.cv_group_report(labels, k=2)
    assert r["n_groups"] == 2
    assert r["n_trainable"] == 3
    assert r["sizes"] == [1, 2]


def test_cv_group_report_excludes_trash_only_videos():
    """A video with nothing but TRASH supplies no training rows, so it is not a group."""
    labels = {_wid("20230118_34", 1, 0): "Focused",
              _wid("20220716_44", 3, 0): "TRASH",
              _wid("20220716_44", 3, 1): "TRASH"}
    r = ar.cv_group_report(labels, k=2)
    assert r["n_groups"] == 1
    assert r["n_trainable"] == 1


def test_cv_group_report_counts_videos_without_the_minority_class():
    labels = {_wid("20230118_34", 1, 0): "Distracted",
              _wid("20220716_44", 3, 0): "Focused",
              _wid("20220913_42", 5, 0): "Focused"}
    r = ar.cv_group_report(labels, k=2)
    assert r["groups_without_distracted"] == 2


def test_cv_group_report_folds_partition_every_window():
    labels = {_wid(v, c, 0): ("Distracted" if c % 2 else "Focused")
              for v in ("20230118_34", "20220716_44", "20220913_42", "20201015_10")
              for c in range(4)}
    r = ar.cv_group_report(labels, k=4)
    assert len(r["folds"]) == 4
    assert sum(f["n"] for f in r["folds"]) == r["n_trainable"]
    assert sum(f["distracted"] for f in r["folds"]) == 8
    assert sum(f["videos"] for f in r["folds"]) == r["n_groups"]   # no video in two folds


def test_cv_group_report_empty_when_nothing_trainable():
    r = ar.cv_group_report({_wid("20230118_34", 1, 0): "TRASH"}, k=5)
    assert r["n_groups"] == 0 and r["folds"] == []


def test_main_prints_fold_balance(tmp_path, capsys):
    p = tmp_path / "annotations.csv"
    p.write_text("window_id,label\n" + "".join(
        f"{_wid(v, c, 0)},{'Distracted' if c else 'Focused'}\n"
        for v in ("20230118_34", "20220716_44") for c in range(2)))

    ar.main(annotations_csv=str(p), pbtxt=None, folds=2)

    out = capsys.readouterr().out
    assert "CV grouping" in out
    assert "GroupKFold(2)" in out
