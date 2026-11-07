# SAV dataset

The working dataset. Layout under `datasets/sav/`:

| path | contents |
|------|----------|
| `videos/` | raw SAV source `.mp4` (drop them here; gitignored) |
| `pose_tracks/` | `*_timeline.jsonl` from `extract_poses.py` (gitignored) |
| `windows/` | `*_windows.pt` from `make_windows.py` (gitignored) |
| `windows_filtered/` | quality-filtered windows — **label + train on these** (gitignored) |
| `annotations/` | SAV's own AVA action labels — `train.csv`, `val.csv`, `education_first_label.pbtxt` (third-party; gitignored) |
| `calibration_set.txt` | shared annotator-agreement window list (tracked) |
| `annotation_worklist.txt` | ordered windows a human should label (tracked) |
| `shards/` | per-annotator disjoint worklists (tracked) |
| `sav_dataset.txt` | source video URL list (tracked) |

## Run order

Every step defaults to the paths above (from `lib/paths.py`), so after dropping
videos into `videos/` you can run each with no flags:

1. `python steps/extract_poses.py`          # videos/ → pose_tracks/
2. `python steps/make_windows.py`           # pose_tracks/ → windows/
3. `python steps/filter_windows.py`         # windows/ → windows_filtered/
4. `python steps/build_annotation_set.py`   # → annotation_worklist.txt + calibration_set.txt + shards/
5. Calibration round — all 3 annotators label the SAME calibration subset, each into
   their OWN csv (annotate.py keys rows by window_id, so a shared file would overwrite):
   `python steps/annotate.py --calibration calibration_set.txt --annotations datasets/sav/calibration/annotator_1.csv`
6. Gate on agreement (κ ≥ 0.6), pairwise across the three annotator CSVs:
   `python steps/annotation_report.py --annotations <a>.csv --annotations-b <b>.csv`
7. Mass labeling — each annotator takes their own shard into the shared ledger:
   `python steps/annotate.py --calibration datasets/sav/shards/annotator_1.txt`

Labels are written to `datasets/annotations.csv` (tracked, shared).
