"""diagnose_representation.py -- the measurements behind 05_representation.md.

These pin the parts that are easy to get subtly wrong and impossible to notice
afterwards: which slice of the 85-d vector each feature group names, and the
leave-one-out rule that stops the human ceiling from scoring an annotator
against a consensus they themselves voted for.
"""
import numpy as np
import pytest
import torch

import baselines as bl
import diagnose_representation as dr


class TestFeatureGroups:
    def test_slices_match_pose_stat_features_layout(self):
        """mean(2x17) then std(2x17) then speed(17) -- 85 in that order."""
        feat = bl.pose_stat_features(torch.zeros(3, 90, 17))
        assert feat.shape[0] == 85
        by_name = {n: (lo, hi) for n, lo, hi in dr.FEATURE_GROUPS}
        assert by_name["mean only (static posture)"] == (0, 34)
        assert by_name["std only (per-joint motion spread)"] == (34, 68)
        assert by_name["speed only (mean joint speed)"] == (68, 85)

    def test_every_group_stays_inside_the_vector(self):
        for _, lo, hi in dr.FEATURE_GROUPS:
            assert 0 <= lo < hi <= 85

    def test_ablation_reports_one_row_per_group(self):
        rng = np.random.default_rng(0)
        n = 120
        y = (rng.random(n) < 0.35).astype(np.int64)
        # Signal lives only in the std block, so the groups must not tie.
        feats = rng.normal(0, 1, (n, 85)).astype(np.float32)
        feats[:, 34:68] += y[:, None] * 2.0
        folds = [(np.arange(n)[i::3], np.arange(n)[(i + 1) % 3::3])
                 for i in range(3)]
        rows = dr.feature_group_ablation(feats, y, folds, 2)
        assert len(rows) == len(dr.FEATURE_GROUPS)
        by = {r["name"]: r["macro_f1"] for r in rows}
        assert (by["std only (per-joint motion spread)"]
                > by["mean only (static posture)"])


class TestJointConfidence:
    def test_reads_channel_two_and_names_coco17(self):
        w = torch.zeros(3, 90, 17)
        w[2, :, :] = 0.9
        w[2, :, 15] = 0.1                      # ankle_L unobserved
        rows = dr.joint_confidence([w])
        assert [r["joint"] for r in rows] == dr.COCO17
        assert rows[15]["joint"] == "ankle_L"
        assert rows[15]["observed"] is False
        assert rows[5]["observed"] is True

    def test_threshold_is_the_documented_one(self):
        w = torch.zeros(3, 90, 17)
        w[2, :, :] = dr.UNOBSERVED_CONF + 1e-4
        assert all(r["observed"] for r in dr.joint_confidence([w]))
        w[2, :, :] = dr.UNOBSERVED_CONF - 1e-4
        assert not any(r["observed"] for r in dr.joint_confidence([w]))


class TestHumanCeiling:
    def _ledgers(self, rows):
        out = {}
        for wid, a, b, c in rows:
            out[wid] = {"annotator_1": a, "annotator_2": b, "annotator_3": c}
        return out

    def test_scores_a_judge_only_where_the_other_two_agree(self):
        """The judge's own vote must never help define the target."""
        led = self._ledgers([
            ("w1", "Focused", "Focused", "Focused"),
            ("w2", "Distracted", "Focused", "Focused"),   # 1 differs from 2&3
            ("w3", "Focused", "Distracted", "Focused"),   # 2 differs, 1&3 agree
        ])
        hc = dr.human_ceiling(led)
        by = {a["annotator"]: a for a in hc["per_annotator"]}
        # annotator_1 is scored on w1 and w3 (where 2 and 3 agree)... on w3 they
        # disagree, so 2 and 3 do NOT agree there; only w1 and w2 qualify.
        assert by["annotator_1"]["n"] == 2
        assert by["annotator_1"]["accuracy"] == pytest.approx(0.5)

    def test_drops_windows_anyone_called_trash(self):
        """A TRASH window has no Focused/Distracted truth to be scored against."""
        led = self._ledgers([
            ("w1", "Focused", "Focused", "Focused"),
            ("w2", "TRASH", "Focused", "Focused"),
        ])
        hc = dr.human_ceiling(led)
        assert hc["n_all_three_trainable"] == 1

    def test_ignores_windows_only_one_person_saw(self):
        led = {"w1": {"annotator_1": "Focused"},
               "w2": {"annotator_1": "Focused", "annotator_2": "Focused",
                      "annotator_3": "Focused"}}
        hc = dr.human_ceiling(led)
        assert hc["n_overlap_windows"] == 1
        assert hc["n_all_three_trainable"] == 1

    def test_perfect_agreement_gives_a_perfect_score(self):
        led = self._ledgers([(f"w{i}", "Focused", "Focused", "Focused")
                             for i in range(5)]
                            + [(f"d{i}", "Distracted", "Distracted", "Distracted")
                               for i in range(5)])
        hc = dr.human_ceiling(led)
        assert hc["mean_accuracy"] == pytest.approx(1.0)
        assert hc["mean_macro_f1"] == pytest.approx(1.0)

    def test_survives_an_empty_overlap(self):
        hc = dr.human_ceiling({"w1": {"annotator_1": "Focused"}})
        assert hc["per_annotator"] == []
        assert np.isnan(hc["mean_macro_f1"])
