"""Geometric augmentation for config C.

All transforms operate on windows that common.prenormalize_2d has ALREADY
normalized, so masked (low-confidence) joints sit at exactly (0, 0). Every
transform must leave those at (0, 0): reviving a masked joint invents a
coordinate where the pipeline recorded "unknown".
"""
import torch

import augment
import model
import pytest


def _window(T=90, V=17):
    """A window with distinctive per-joint values so permutations are visible."""
    w = torch.zeros(3, T, V)
    for v in range(V):
        w[0, :, v] = float(v + 1)          # x
        w[1, :, v] = float(100 + v)        # y
        w[2, :, v] = 0.9                   # conf
    return w


class TestRotate:
    def test_zero_degrees_is_identity(self):
        w = _window()
        torch.testing.assert_close(augment.rotate(w, 0.0), w)

    def test_ninety_degrees_maps_x_to_y(self):
        w = torch.zeros(3, 1, 17)
        w[0, 0, 0] = 1.0      # x = 1
        w[1, 0, 0] = 0.0      # y = 0
        out = augment.rotate(w, 90.0)
        assert abs(float(out[0, 0, 0]) - 0.0) < 1e-6
        assert abs(float(out[1, 0, 0]) - 1.0) < 1e-6

    def test_preserves_distance_from_origin(self):
        w = _window()
        out = augment.rotate(w, 37.0)
        before = (w[0] ** 2 + w[1] ** 2).sqrt()
        after = (out[0] ** 2 + out[1] ** 2).sqrt()
        torch.testing.assert_close(before, after)

    def test_confidence_channel_untouched(self):
        w = _window()
        torch.testing.assert_close(augment.rotate(w, 12.0)[2], w[2])

    def test_masked_joints_stay_at_origin(self):
        w = _window()
        w[0, :, 3] = 0.0
        w[1, :, 3] = 0.0
        out = augment.rotate(w, 15.0)
        assert torch.all(out[0, :, 3] == 0.0)
        assert torch.all(out[1, :, 3] == 0.0)

    def test_does_not_mutate_input(self):
        w = _window()
        before = w.clone()
        augment.rotate(w, 15.0)
        torch.testing.assert_close(w, before)


class TestHflip:
    def test_permutation_is_a_valid_involution(self):
        perm = augment.COCO_FLIP_PERM
        assert len(perm) == 17
        assert sorted(perm) == list(range(17))
        assert [perm[perm[i]] for i in range(17)] == list(range(17))

    def test_swaps_left_and_right_joint_indices(self):
        """COCO-17: 5=left_shoulder, 6=right_shoulder. Mirroring coordinates
        without swapping these produces an anatomically impossible body, and it
        fails silently -- the tensor shape does not change."""
        w = _window()
        out = augment.hflip(w)
        # joint 5 of the output must carry the ORIGINAL joint 6's data
        torch.testing.assert_close(out[1, :, 5], w[1, :, 6])
        torch.testing.assert_close(out[1, :, 6], w[1, :, 5])
        torch.testing.assert_close(out[2, :, 5], w[2, :, 6])

    def test_negates_x(self):
        w = _window()
        out = augment.hflip(w)
        torch.testing.assert_close(out[0, :, 5], -w[0, :, 6])

    def test_flip_twice_is_identity(self):
        w = _window()
        torch.testing.assert_close(augment.hflip(augment.hflip(w)), w)

    def test_masked_joints_stay_at_origin(self):
        w = _window()
        w[0, :, 5] = 0.0
        w[1, :, 5] = 0.0
        out = augment.hflip(w)
        assert torch.all(out[0, :, 6] == 0.0)
        assert torch.all(out[1, :, 6] == 0.0)

    def test_does_not_mutate_input(self):
        w = _window()
        before = w.clone()
        augment.hflip(w)
        torch.testing.assert_close(w, before)


class TestTemporalCrop:
    def test_output_keeps_the_original_length(self):
        w = _window(T=90)
        out = augment.temporal_crop(w, start=10, length=72)
        assert out.shape == (3, 90, 17)

    def test_full_length_crop_is_identity(self):
        w = _window(T=90)
        torch.testing.assert_close(augment.temporal_crop(w, start=0, length=90), w)

    def test_crop_selects_the_requested_span(self):
        """A ramp in time lets us see exactly which span survived."""
        w = torch.zeros(3, 90, 17)
        w[0, :, 0] = torch.arange(90, dtype=torch.float32)
        out = augment.temporal_crop(w, start=30, length=30)
        # the cropped span is frames 30..59, stretched over 90 output frames
        assert abs(float(out[0, 0, 0]) - 30.0) < 1e-4
        assert abs(float(out[0, -1, 0]) - 59.0) < 1e-4

    def test_confidence_channel_is_resampled_too(self):
        w = _window(T=90)
        w[2, :, 0] = torch.linspace(0.0, 1.0, 90)
        out = augment.temporal_crop(w, start=0, length=45)
        assert out[2, :, 0].shape[0] == 90
        assert float(out[2, -1, 0]) < 0.55   # only the first half survived

    def test_does_not_mutate_input(self):
        w = _window()
        before = w.clone()
        augment.temporal_crop(w, start=5, length=70)
        torch.testing.assert_close(w, before)


class TestSkeletonAugment:
    def test_registered_defaults_are_the_pre_registered_magnitudes(self):
        """These are fixed a priori (design doc Sec. 2.3). Changing them after
        seeing fold results would be tuning on the test set."""
        assert augment.DEFAULT_AUGMENT == {
            "max_rotation_deg": 15.0, "flip_prob": 0.5, "min_crop_ratio": 0.8}

    def test_output_shape_is_unchanged(self):
        aug = augment.SkeletonAugment(seed=0)
        out = aug(_window())
        assert out.shape == (3, 90, 17)

    def test_it_actually_changes_the_window(self):
        aug = augment.SkeletonAugment(seed=0)
        w = _window()
        assert not torch.equal(aug(w), w)

    def test_same_seed_gives_the_same_result(self):
        w = _window()
        a = augment.SkeletonAugment(seed=123)(w)
        b = augment.SkeletonAugment(seed=123)(w)
        torch.testing.assert_close(a, b)

    def test_different_seeds_give_different_results(self):
        w = _window()
        a = augment.SkeletonAugment(seed=1)(w)
        b = augment.SkeletonAugment(seed=2)(w)
        assert not torch.equal(a, b)

    def test_successive_calls_differ(self):
        """Fresh randomness per sample, per epoch -- not one fixed transform."""
        aug = augment.SkeletonAugment(seed=7)
        w = _window()
        assert not torch.equal(aug(w), aug(w))

    def test_rotation_stays_within_the_registered_bound(self):
        """Rotating a [-1,1] box by 15 degrees reaches at most ~1.10 in each
        axis. The design deliberately does NOT re-normalize afterwards
        (prenormalize_2d guards on a raw-pixel extent > 10 and would silently
        skip the rescale), so the expansion is bounded and asserted instead."""
        w = torch.zeros(3, 90, 17)
        w[0] = 1.0
        w[1] = 1.0
        w[2] = 0.9
        aug = augment.SkeletonAugment(seed=0)
        for _ in range(50):
            out = aug(w)
            assert float(out[0].abs().max()) <= 1.45
            assert float(out[1].abs().max()) <= 1.45

    def test_crop_never_goes_below_the_registered_floor(self):
        """Recorded crop ratios must all be >= 0.8: below 2/3 the crop could
        drop SAV's keyframe at frame 30 of 90."""
        aug = augment.SkeletonAugment(seed=5)
        for _ in range(200):
            assert aug._draw_crop_ratio() >= 0.8

    def test_one_rotation_angle_per_window_not_per_frame(self):
        """A per-frame angle would simulate a shaking camera rather than a
        differently-mounted one, injecting temporal noise the model then has to
        learn to ignore.

        Detected without inspecting internals: disable flip and crop, then check
        that every frame's joint-0 vector kept the same angle relative to the
        original. A per-frame draw would make those angles differ.
        """
        w = torch.zeros(3, 90, 17)
        w[0, :, 0] = 1.0
        w[1, :, 0] = 0.0
        w[2] = 0.9
        aug = augment.SkeletonAugment(flip_prob=0.0, min_crop_ratio=1.0, seed=3)
        out = aug(w)
        angles = torch.atan2(out[1, :, 0], out[0, :, 0])
        assert float(angles.max() - angles.min()) < 1e-5


class TestAugmentedTensorDataset:
    def test_without_augment_it_returns_the_raw_tensors(self):
        X = torch.stack([_window(), _window()])
        y = torch.tensor([0, 1])
        ds = augment.AugmentedTensorDataset(X, y, augment=None)
        assert len(ds) == 2
        x0, y0 = ds[0]
        torch.testing.assert_close(x0, X[0])
        assert int(y0) == 0

    def test_with_augment_the_sample_is_transformed(self):
        X = torch.stack([_window(), _window()])
        y = torch.tensor([0, 1])
        ds = augment.AugmentedTensorDataset(
            X, y, augment=augment.SkeletonAugment(seed=0))
        x0, y0 = ds[0]
        assert not torch.equal(x0, X[0])
        assert int(y0) == 0, "the label must survive augmentation unchanged"

    def test_underlying_tensors_are_not_mutated(self):
        X = torch.stack([_window(), _window()])
        before = X.clone()
        ds = augment.AugmentedTensorDataset(
            X, torch.tensor([0, 1]), augment=augment.SkeletonAugment(seed=0))
        _ = ds[0]
        torch.testing.assert_close(X, before)


class TestFlipPermForASubset:
    """Rung E1 trains on 13 joints, so the mirror has to be reindexed with them.

    Applying the COCO-17 permutation to a 13-node window would index past the
    end of the tensor -- or, for a subset that happened to be long enough, swap
    the wrong pair.
    """

    def test_full_skeleton_gives_the_coco_permutation(self):
        assert augment.flip_perm_for(list(range(17))) == augment.COCO_FLIP_PERM

    def test_observed_subset_swaps_the_surviving_pairs(self):
        assert augment.flip_perm_for(model.OBSERVED_JOINTS_13) == \
            [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]

    def test_result_is_an_involution_over_the_subset(self):
        perm = augment.flip_perm_for(model.OBSERVED_JOINTS_13)
        assert [perm[perm[i]] for i in range(len(perm))] == list(range(len(perm)))

    def test_refuses_a_subset_that_keeps_only_one_side(self):
        """[0, 1] keeps the left eye but not the right, so there is nothing for
        it to mirror onto. Silently mapping it to itself would teach the model
        that a mirrored left eye is a left eye."""
        with pytest.raises(ValueError):
            augment.flip_perm_for([0, 1])


class TestHflipOnASubsetWindow:
    def test_swaps_shoulders_of_a_13_joint_window(self):
        w = _window(V=13)
        out = augment.hflip(w, flip_perm=augment.flip_perm_for(model.OBSERVED_JOINTS_13))
        torch.testing.assert_close(out[1, :, 5], w[1, :, 6])
        torch.testing.assert_close(out[1, :, 6], w[1, :, 5])

    def test_default_perm_is_still_the_coco_one(self):
        w = _window()
        torch.testing.assert_close(augment.hflip(w), augment.hflip(w, flip_perm=None))


class TestSkeletonAugmentOnASubsetWindow:
    def test_keeps_the_subset_node_count(self):
        aug = augment.SkeletonAugment(
            seed=0, flip_perm=augment.flip_perm_for(model.OBSERVED_JOINTS_13))
        assert aug(_window(V=13)).shape == (3, 90, 13)

    def test_default_run_is_byte_identical_to_before_the_flag(self):
        """Config C's augmentation must not move: E1 is compared against it."""
        w = _window()
        torch.testing.assert_close(augment.SkeletonAugment(seed=5)(w),
                                   augment.SkeletonAugment(seed=5, flip_perm=None)(w))


class TestAugmentedTensorDatasetTransform:
    """Rung E3 converts to bone vectors AFTER augmenting.

    Order matters: rotating a skeleton and then taking differences is the same
    geometry the test fold sees, whereas augmenting already-differenced data
    would crop and flip quantities that are no longer joint positions.
    """

    def test_transform_runs_after_the_augmentation(self):
        data = augment.AugmentedTensorDataset(
            torch.zeros(1, 3, 4, 17), torch.zeros(1, dtype=torch.long),
            augment=lambda w: w + 1.0, transform=lambda w: w * 10.0)
        x, _ = data[0]
        assert float(x[0, 0, 0]) == 10.0      # (0 + 1) * 10, not 0 * 10 + 1

    def test_transform_applies_without_augmentation(self):
        data = augment.AugmentedTensorDataset(
            torch.ones(1, 3, 4, 17), torch.zeros(1, dtype=torch.long),
            transform=lambda w: w * 3.0)
        assert float(data[0][0][0, 0, 0]) == 3.0

    def test_dataset_without_either_is_unchanged(self):
        x = torch.rand(2, 3, 4, 17)
        data = augment.AugmentedTensorDataset(x, torch.zeros(2, dtype=torch.long))
        torch.testing.assert_close(data[1][0], x[1])
