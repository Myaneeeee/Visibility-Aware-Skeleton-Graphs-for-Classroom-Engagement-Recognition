import torch
import transforms


def test_resample_length_stretches_whole_span_to_out_len():
    """A clip-span window covers the entire clip, so its span maps onto the full
    output with no clamped tail."""
    x = torch.arange(54, dtype=torch.float32).reshape(1, 54, 1).expand(3, 54, 17).contiguous()

    out = transforms.resample_length(x, out_len=90)

    assert out.shape == (3, 90, 17)
    assert out[0, 0, 0] == 0.0
    assert out[0, -1, 0] == 53.0                      # last input frame, not clamped early
    assert torch.all(out[0, 1:, 0] > out[0, :-1, 0])  # strictly increasing: no freeze


def test_resample_length_is_identity_at_target():
    x = torch.rand(3, 90, 17)
    out = transforms.resample_length(x, out_len=90)
    assert torch.allclose(out, x)




class TestToBone:
    """Rung E3's input: each joint as a vector from its parent.

    Section 3.3 measured bone vectors as the best representation of the twelve
    tried (0.8184 linear, against 0.7926 for the 85 per-joint statistics). The
    graph model has only ever been given raw coordinates.
    """

    def _window(self, T=4, V=17):
        w = torch.zeros(3, T, V)
        for v in range(V):
            w[0, :, v] = float(v + 1)
            w[1, :, v] = float(100 + v)
            w[2, :, v] = 0.5 + v / 100.0
        return w

    def test_shape_is_unchanged(self):
        import model
        w = self._window()
        assert transforms.to_bone(w, model.bone_parents()).shape == (3, 4, 17)

    def test_root_bone_is_the_zero_vector(self):
        """The nose has no parent, so it carries no displacement."""
        import model
        out = transforms.to_bone(self._window(), model.bone_parents())
        assert torch.all(out[0, :, 0] == 0.0)
        assert torch.all(out[1, :, 0] == 0.0)

    def test_bone_is_the_joint_minus_its_parent(self):
        import model
        w = self._window()
        out = transforms.to_bone(w, model.bone_parents())
        # joint 7 (elbow_L) hangs off joint 5 (shoulder_L)
        torch.testing.assert_close(out[0, :, 7], w[0, :, 7] - w[0, :, 5])
        torch.testing.assert_close(out[1, :, 7], w[1, :, 7] - w[1, :, 5])

    def test_confidence_is_the_weaker_of_the_two_endpoints(self):
        """A bone is only as observed as its worse end. Keeping the child's own
        confidence would claim a well-seen hip-to-knee bone off an unseen knee."""
        import model
        w = self._window()
        w[2, :, 5] = 0.2          # shoulder barely seen
        out = transforms.to_bone(w, model.bone_parents())
        torch.testing.assert_close(out[2, :, 7], torch.full((4,), 0.2))

    def test_root_keeps_its_own_confidence(self):
        import model
        w = self._window()
        out = transforms.to_bone(w, model.bone_parents())
        torch.testing.assert_close(out[2, :, 0], w[2, :, 0])

    def test_translating_the_skeleton_leaves_bones_unchanged(self):
        """The property that makes bones worth trying: they are already
        translation-invariant, so the model does not spend capacity on it."""
        import model
        w = self._window()
        shifted = w.clone()
        shifted[0] += 7.0
        shifted[1] -= 3.0
        parents = model.bone_parents()
        torch.testing.assert_close(transforms.to_bone(w, parents),
                                   transforms.to_bone(shifted, parents))

    def test_does_not_mutate_its_input(self):
        import model
        w = self._window()
        before = w.clone()
        transforms.to_bone(w, model.bone_parents())
        torch.testing.assert_close(w, before)

    def test_works_on_a_subset_window(self):
        import model
        w = self._window(V=13)
        parents = model.bone_parents(joints=model.OBSERVED_JOINTS_13)
        out = transforms.to_bone(w, parents)
        assert out.shape == (3, 4, 13)
        torch.testing.assert_close(out[0, :, 11], w[0, :, 11] - w[0, :, 5])
