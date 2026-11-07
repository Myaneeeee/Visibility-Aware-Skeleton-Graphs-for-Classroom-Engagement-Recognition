# NTU-2D Pretraining Data (local-only)

This directory holds the NTU RGB+D 60 2D-pose pickle and the exported ST-GCN
backbone. Both are gitignored (multi-GB / regenerable).

## 1. Download `ntu60_hrnet.pkl`

From the pyskl model/data zoo (OpenMMLab). Record the exact URL + SHA256 + size
you used here after downloading:

- URL: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl
- SHA256: 953836d9dcae0b139ba524cee5003346c5cb4cbb26b0371cbf927f0cfa383c6a
- Size: 705,401,771 bytes (673 MiB)

Place it at `datasets/ntu/ntu60_hrnet.pkl` (the default in `lib/paths.py`).

## 2. Sanity-check conversion (fast)

```bash
.venv/bin/python -c "import _bootstrap, ntu_dataset as nd; a=nd.load_split('datasets/ntu/ntu60_hrnet.pkl', limit=4); print(len(a), nd.anno_to_window(a[0]).shape)"
```

Expect `4 torch.Size([3, 90, 17])`.

## 3. Smoke pretrain (a few minutes, verifies the pipeline)

```bash
.venv/bin/python steps/pretrain_ntu.py --limit 512 --epochs 2
```

## 4. Full pretrain (~1 hour on RTX 3050; X-Sub train split)

```bash
.venv/bin/python steps/pretrain_ntu.py --epochs 30
```

Both commands default `--pkl` to `datasets/ntu/ntu60_hrnet.pkl` and `--out` to
`datasets/ntu/stgcn_ntu60_backbone.pth` — consumed by Plan 3 via
`transfer.load_pretrained_backbone(model, "datasets/ntu/stgcn_ntu60_backbone.pth")`.
Report the final NTU60 train/val loss and top-1 accuracy honestly (printed at the
end as "NTU60 train top-1 @ best val") — sanity that the backbone learned motion
features, not a SOTA claim.
