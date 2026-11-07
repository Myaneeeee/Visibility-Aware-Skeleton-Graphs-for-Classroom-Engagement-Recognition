import build_annotation_set as bas


def _wids(spec):
    """spec: {video_id: n_windows} -> window ids across clip 1 of each video."""
    out = []
    for vid, n in spec.items():
        for i in range(n):
            out.append(f"{vid}_1_id_{i}_f0_f74")
    return out


def test_assign_gives_every_window_to_someone():
    wids = _wids({"20230101_01": 20, "20230102_02": 20, "20230103_03": 20})
    lists, overlap = bas.assign(wids, n_annotators=3, overlap=0.10, seed=42)
    covered = set().union(*[set(l) for l in lists])
    assert covered == set(wids)


def test_overlap_windows_go_to_every_annotator():
    wids = _wids({"20230101_01": 20, "20230102_02": 20, "20230103_03": 20})
    lists, overlap = bas.assign(wids, n_annotators=3, overlap=0.10, seed=42)
    assert len(overlap) == 6
    for lst in lists:
        assert set(overlap) <= set(lst)


def test_non_overlap_windows_go_to_exactly_one_annotator():
    wids = _wids({"20230101_01": 20, "20230102_02": 20, "20230103_03": 20})
    lists, overlap = bas.assign(wids, n_annotators=3, overlap=0.10, seed=42)
    overlap = set(overlap)
    for wid in set(wids) - overlap:
        assert sum(1 for l in lists if wid in l) == 1


def test_total_judgements_matches_the_overlap_formula():
    wids = _wids({"20230101_01": 30, "20230102_02": 30})
    lists, overlap = bas.assign(wids, n_annotators=3, overlap=0.10, seed=42)
    assert sum(len(l) for l in lists) == len(wids) + 2 * len(overlap)


def test_overlap_is_shuffled_into_each_list_not_appended():
    """Annotators must not be able to spot the cross-checked windows by position."""
    wids = _wids({"20230101_01": 50, "20230102_02": 50})
    lists, overlap = bas.assign(wids, n_annotators=3, overlap=0.10, seed=42)
    overlap = set(overlap)
    for lst in lists:
        positions = [i for i, w in enumerate(lst) if w in overlap]
        assert max(positions) > len(lst) / 2
        assert min(positions) < len(lst) / 2


def test_each_annotator_sees_every_video():
    """Assignment spreads across videos so annotator is not confounded with video."""
    wids = _wids({f"2023010{i}_0{i}": 30 for i in range(1, 6)})
    lists, _ = bas.assign(wids, n_annotators=3, overlap=0.10, seed=42)
    for lst in lists:
        assert len(bas.group_by_source(lst)) == 5


def test_assign_is_deterministic():
    wids = _wids({"20230101_01": 20, "20230102_02": 20})
    a, _ = bas.assign(wids, 3, 0.10, seed=7)
    b, _ = bas.assign(wids, 3, 0.10, seed=7)
    assert a == b


def test_spread_across_videos_alternates_sources():
    wids = _wids({"20230101_01": 3, "20230102_02": 3, "20230103_03": 3})
    ordered = bas.spread_across_videos(wids, seed=1)
    sources = [w.split("_1_id_")[0] for w in ordered]
    assert len(set(sources[:3])) == 3


def test_zero_overlap_produces_disjoint_lists():
    wids = _wids({"20230101_01": 30, "20230102_02": 30})
    lists, overlap = bas.assign(wids, n_annotators=3, overlap=0.0, seed=42)
    assert overlap == []
    assert sum(len(l) for l in lists) == len(wids)
