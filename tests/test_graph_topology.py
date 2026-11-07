"""Graph topology for the E-ladder: an adjacency built over the observed joints.

Mean detector confidence over every frame of every labeled window puts knees at
0.29 and ankles at 0.19, far under the 0.35 usability threshold -- students sit
behind desks. Rung E1 rebuilds the graph over the 13 joints that are actually
observed. The guard that matters most here is the last class: with nothing
dropped, the matrix must be identical to the one configs A/B/C trained on, or
the E-vs-C comparison is measuring the refactor instead of the topology.
"""
import json
import os

import pytest
import torch

import numpy as np

import model as model_mod
import paths
from model import (OBSERVED_JOINTS_13, STGCNClassifier,
                   build_spatial_partition_matrix, get_coco_edges)

ALL_JOINTS = list(range(17))


def _support(A):
    """Where the graph has any edge at all, across the three partitions."""
    return (A.abs().sum(dim=0) > 0)


class TestObservedJointSet:
    def test_drops_exactly_the_knees_and_ankles(self):
        assert sorted(set(ALL_JOINTS) - set(OBSERVED_JOINTS_13)) == [13, 14, 15, 16]

    def test_keeps_thirteen_joints(self):
        assert len(OBSERVED_JOINTS_13) == 13

    def test_matches_the_measured_observability(self):
        """The subset is a measurement, not a preference: it must agree with
        what diagnose_representation.py recorded."""
        f = os.path.join(paths.diagnostics_dir(), "representation.json")
        if not os.path.exists(f):
            pytest.skip("representation.json not generated on this machine")
        with open(f) as fh:
            measured = json.load(fh)["joint_confidence"]
        observed = [j["index"] for j in measured if j["observed"]]
        assert sorted(OBSERVED_JOINTS_13) == sorted(observed)


class TestEdgeList:
    def test_defaults_to_the_full_coco_skeleton(self):
        assert len(get_coco_edges()) == 18

    def test_subset_drops_every_edge_touching_a_dropped_joint(self):
        edges = get_coco_edges(joints=OBSERVED_JOINTS_13)
        assert all(i < 13 and j < 13 for i, j in edges)

    def test_subset_keeps_every_edge_between_surviving_joints(self):
        kept = {(i, j) for i, j in get_coco_edges()
                if i in OBSERVED_JOINTS_13 and j in OBSERVED_JOINTS_13}
        assert len(get_coco_edges(joints=OBSERVED_JOINTS_13)) == len(kept)

    def test_reindexes_onto_the_subset_positions(self):
        """Joints are renumbered to their position in the subset, so a graph over
        [0, 5, 6] has edges among nodes 0, 1, 2 -- not 0, 5, 6."""
        assert sorted(get_coco_edges(joints=[0, 5, 6])) == [(0, 1), (0, 2), (1, 2)]


class TestSubsetAdjacency:
    def test_shape_follows_the_subset(self):
        A = build_spatial_partition_matrix(joints=OBSERVED_JOINTS_13)
        assert A.shape == (3, 13, 13)

    def test_hip_keeps_only_its_surviving_neighbours(self):
        """Hip_L (11) connects to shoulder_L, hip_R and knee_L in COCO-17. With
        the knee gone it becomes a leaf of the torso -- the whole point of E1."""
        A = build_spatial_partition_matrix(joints=OBSERVED_JOINTS_13)
        neighbours = {int(j) for j in _support(A)[11].nonzero().flatten()}
        assert neighbours == {5, 11, 12}

    def test_every_surviving_edge_is_present(self):
        A = build_spatial_partition_matrix(joints=OBSERVED_JOINTS_13)
        support = _support(A)
        for i, j in get_coco_edges(joints=OBSERVED_JOINTS_13):
            assert support[i, j] and support[j, i]

    def test_partitions_stay_normalized(self):
        A = build_spatial_partition_matrix(joints=OBSERVED_JOINTS_13)
        sums = A.sum(dim=2)
        assert torch.all((sums - 1.0).abs().lt(1e-6) | sums.abs().lt(1e-6))

    def test_rejects_a_joint_outside_coco_17(self):
        with pytest.raises(ValueError):
            build_spatial_partition_matrix(joints=[0, 5, 17])

    def test_rejects_duplicate_joints(self):
        with pytest.raises(ValueError):
            build_spatial_partition_matrix(joints=[0, 5, 5])

    def test_rejects_a_subset_missing_the_center_node(self):
        """Partitioning is defined by distance to the center joint, so dropping
        it would silently change what partitions 1 and 2 mean."""
        with pytest.raises(ValueError):
            build_spatial_partition_matrix(joints=[5, 6, 7])


class TestFullGraphIsUnchanged:
    """E0-reproduces-C. Every assertion here describes the matrix A/B/C used."""

    def test_default_is_the_full_coco_17_graph(self):
        assert build_spatial_partition_matrix().shape == (3, 17, 17)

    def test_passing_every_joint_is_a_no_op(self):
        assert torch.equal(build_spatial_partition_matrix(),
                           build_spatial_partition_matrix(joints=ALL_JOINTS))

    def test_partition_zero_is_the_self_loop_identity(self):
        assert torch.equal(build_spatial_partition_matrix()[0], torch.eye(17))

    def test_partitions_stay_normalized(self):
        sums = build_spatial_partition_matrix().sum(dim=2)
        assert torch.all((sums - 1.0).abs().lt(1e-6) | sums.abs().lt(1e-6))

    def test_every_coco_edge_is_present_in_one_direction_partition(self):
        support = _support(build_spatial_partition_matrix())
        for i, j in get_coco_edges():
            assert support[i, j] and support[j, i]

    def test_center_node_has_no_centripetal_neighbours(self):
        """Nothing is closer to the center than the center, so partition 1's
        nose row is empty -- the case the row-sum guard has to tolerate."""
        assert float(build_spatial_partition_matrix()[1, 0].sum()) == 0.0


class TestClassifierAcceptsASubsetGraph:
    def test_forwards_a_thirteen_joint_window(self):
        m = STGCNClassifier(num_classes=2, joints=OBSERVED_JOINTS_13)
        assert m(torch.rand(2, 3, 90, 13)).shape == (2, 2)

    def test_default_still_forwards_a_seventeen_joint_window(self):
        assert STGCNClassifier(num_classes=2)(torch.rand(2, 3, 90, 17)).shape == (2, 2)

    def test_subset_model_carries_the_subset_graph(self):
        m = STGCNClassifier(num_classes=2, joints=OBSERVED_JOINTS_13)
        assert m.layer1.gcn.A.shape == (3, 13, 13)


class TestAgainstTheFrozenCheckpoint:
    """The NTU backbone was written before the subset refactor existed.

    Its saved adjacency buffers are therefore a frozen copy of the matrix configs
    A/B/C actually trained on -- the only reference on disk that a refactor
    cannot have touched.
    """

    def test_built_graph_matches_the_one_stored_in_the_ntu_backbone(self):
        if not os.path.exists(paths.NTU_BACKBONE_FILE):
            pytest.skip("NTU backbone not present on this machine")
        ckpt = torch.load(paths.NTU_BACKBONE_FILE, map_location="cpu",
                          weights_only=True)
        stored = [v for k, v in ckpt.items() if k.endswith("gcn.A")]
        assert stored, "checkpoint has no adjacency buffers to compare against"
        A = build_spatial_partition_matrix()
        assert all(torch.equal(s, A) for s in stored)


class TestObservabilityWeightedAdjacency:
    """Rung E2: keep all 17 nodes, but weight each edge by how well the detector
    actually sees its endpoints. A soft prior where E1 makes a hard cut.
    """

    def test_fixed_confidences_agree_with_the_diagnostics_file(self):
        f = os.path.join(paths.diagnostics_dir(), "representation.json")
        if not os.path.exists(f):
            pytest.skip("representation.json not generated on this machine")
        with open(f) as fh:
            measured = json.load(fh)["joint_confidence"]
        for j in measured:
            assert model_mod.COCO_JOINT_CONFIDENCE[j["index"]] == \
                pytest.approx(j["mean_confidence"], abs=0.002)

    def test_weighting_does_not_change_the_shape(self):
        A = build_spatial_partition_matrix(
            weights=model_mod.COCO_JOINT_CONFIDENCE)
        assert A.shape == (3, 17, 17)

    def test_uniform_weights_reproduce_the_unweighted_graph(self):
        """Guard: weighting is a pure extension, off by default."""
        assert torch.equal(build_spatial_partition_matrix(weights=[1.0] * 17),
                           build_spatial_partition_matrix())

    def test_weighting_adds_and_removes_no_edges(self):
        weighted = _support(build_spatial_partition_matrix(
            weights=model_mod.COCO_JOINT_CONFIDENCE))
        assert torch.equal(weighted, _support(build_spatial_partition_matrix()))

    def test_self_loop_partition_stays_the_identity(self):
        """Row 0 has one entry, so normalization cancels any weight on it.

        allclose rather than equal: (1/w)*w costs an ulp in float32, so the
        diagonal lands 6e-8 off 1.0. The off-diagonal is exactly zero.
        """
        A = build_spatial_partition_matrix(
            weights=model_mod.COCO_JOINT_CONFIDENCE)
        assert torch.allclose(A[0], torch.eye(17), atol=1e-6)
        assert float(A[0].sum()) == pytest.approx(17.0)

    def test_rows_stay_normalized(self):
        sums = build_spatial_partition_matrix(
            weights=model_mod.COCO_JOINT_CONFIDENCE).sum(dim=2)
        assert torch.all((sums - 1.0).abs().lt(1e-6) | sums.abs().lt(1e-6))

    def test_a_barely_seen_neighbour_contributes_less(self):
        """Hip_L's row carries both hip_R (0.71) and knee_L (0.29). After
        weighting the knee must send less than the hip does."""
        A = build_spatial_partition_matrix(
            weights=model_mod.COCO_JOINT_CONFIDENCE)
        assert float(A[2, 11, 13]) < float(A[2, 11, 12])

    def test_endpoint_product_equals_source_only_after_normalization(self):
        """Worth asserting because it looks like a bug otherwise: scaling row i
        by conf[i] divides out when the row is normalized, so weighting by
        conf[i]*conf[j] IS weighting each incoming message by its source."""
        conf = model_mod.COCO_JOINT_CONFIDENCE
        product = build_spatial_partition_matrix(weights=conf)
        source_only = model_mod._normalize_partitions(
            model_mod._raw_partitions(list(range(17)), 0)
            * np.asarray(conf, dtype=np.float32)[None, None, :])
        assert torch.allclose(product, torch.tensor(source_only), atol=1e-6)

    def test_rejects_a_weight_vector_of_the_wrong_length(self):
        with pytest.raises(ValueError):
            build_spatial_partition_matrix(weights=[1.0] * 13)

    def test_weights_follow_a_joint_subset(self):
        """Weights are indexed by COCO joint id, not by position in the subset."""
        A = build_spatial_partition_matrix(
            joints=OBSERVED_JOINTS_13, weights=model_mod.COCO_JOINT_CONFIDENCE)
        assert A.shape == (3, 13, 13)


class TestBoneParents:
    """Rung E3 feeds bone vectors: each joint relative to its parent in the
    skeleton tree, rooted at the same center node the partitions use.
    """

    def test_root_is_its_own_parent(self):
        assert model_mod.bone_parents()[0] == 0

    def test_parents_follow_the_skeleton_outward_from_the_nose(self):
        parents = model_mod.bone_parents()
        expected = {1: 0, 2: 0, 5: 0, 6: 0, 3: 1, 4: 2,
                    7: 5, 8: 6, 11: 5, 12: 6,
                    9: 7, 10: 8, 13: 11, 14: 12, 15: 13, 16: 14}
        assert {j: parents[j] for j in expected} == expected

    def test_every_parent_edge_is_a_real_skeleton_edge(self):
        parents = model_mod.bone_parents()
        edges = {frozenset(e) for e in model_mod.COCO_EDGES}
        for child, parent in enumerate(parents):
            if child != parent:
                assert frozenset((child, parent)) in edges

    def test_parent_map_is_mirror_symmetric(self):
        """Left and right limbs must hang the same way, or a horizontal flip
        would change which bone each joint carries."""
        import augment
        parents = model_mod.bone_parents()
        perm = augment.COCO_FLIP_PERM
        assert [perm[parents[perm[j]]] for j in range(17)] == parents

    def test_subset_reindexes_the_parents(self):
        parents = model_mod.bone_parents(joints=OBSERVED_JOINTS_13)
        assert len(parents) == 13
        assert parents[11] == 5 and parents[12] == 6

    def test_subset_orphans_are_rerooted_not_dangling(self):
        """Dropping joint 5 would leave 7 and 11 without their parent. They
        must attach to something inside the subset, not point outside it."""
        parents = model_mod.bone_parents(joints=[0, 6, 7, 8])
        assert all(0 <= p < 4 for p in parents)
