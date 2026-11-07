import numpy as np
import torch


def resample_length(x, out_len=90):
    """Stretch the whole time span of x onto out_len frames.

    x: tensor (C, T, V). Returns (C, out_len, V).
    """
    C, T, V = x.shape
    if T == out_len:
        return x.clone()
    xin = x.numpy()
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, out_len)
    out = np.empty((C, out_len, V), dtype=np.float32)
    for c in range(C):
        for v in range(V):
            out[c, :, v] = np.interp(dst, src, xin[c, :, v])
    return torch.from_numpy(out)


def to_bone(x, parents):
    """Re-express a window as bone vectors: each joint minus its parent.

    x: (3, T, V) of x, y, confidence. Returns the same shape, where channels 0
    and 1 are the displacement from the parent and channel 2 is the weaker of
    the two endpoints' confidences -- a bone is only as observed as its worse
    end. The root's displacement is zero and it keeps its own confidence.
    """
    idx = torch.as_tensor(parents, dtype=torch.long)
    out = x.clone()
    out[0] = x[0] - x[0][:, idx]
    out[1] = x[1] - x[1][:, idx]
    out[2] = torch.minimum(x[2], x[2][:, idx])
    return out
