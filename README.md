# Visibility-Aware Skeleton Graphs for Classroom Engagement Recognition

This project recognizes whether a student in a classroom video is on-task or off-task from
their body pose alone. It asks one question: what happens to a skeleton graph model when the
camera cannot see part of the body?

## The idea

A spatial-temporal graph convolutional network (ST-GCN) treats the body as a graph of joints.
The standard graph has all 17 COCO joints, because it was designed for footage where the whole
body is visible. Classrooms are different: desks hide the legs. The pose detector still outputs
knees and ankles, but they are mostly guesses.

Across our data, all four knee and ankle joints have a mean detector confidence below 0.35, and
all four fall below that level in 73.6% of samples. So we build the graph from only the 13 joints
the camera actually sees, and measure what that changes.

## How it works

1. **Select clips.** Three-second clips from the public Student Action Video (SAV) dataset.
2. **Extract poses.** YOLO11n-pose finds 17 joints per person, and BoT-SORT tracks each person
   across frames.
3. **Build samples.** Each tracked student becomes one window of 90 frames. Windows with broken
   skeletons are filtered out.
4. **Label.** Three annotators mark each window as Focused (on-task), Distracted (off-task), or
   TRASH (unusable pose, or not a student).
5. **Pretrain.** The ST-GCN backbone is pretrained on NTU RGB+D 60.
6. **Evaluate.** Every model is scored with 5-fold cross-validation grouped by source video, so a
   lesson never appears in both training and testing. Each comparison is repeated over seeds
   51 to 60.

## Models

| name | what it is |
|---|---|
| A | ST-GCN trained from scratch on 17 joints |
| B | A with the NTU-60 pretrained backbone |
| C | B with data augmentation |
| E1 | C with the graph rebuilt on the 13 visible joints (ST-GCN-13) |
| E2 | C with graph edges weighted by joint visibility |
| E3 | E1 using bone vectors instead of joint coordinates |
| E5 | C with only the final pooling limited to the 13 joints |
| E6 | C with the 4 hidden joints blocked at the input |
| E7 | C with per-joint input normalization |
| MLP, GRU | baselines that do not use a graph |

Monte Carlo dropout and an MLP + ST-GCN-13 ensemble are evaluated from the saved predictions.

## Dataset

| | |
|---|---|
| source videos | 200 classroom lessons from SAV |
| clips | 1,017 |
| labeled windows | 5,135 |
| rejected as TRASH | 1,860 |
| trainable windows | 3,275 from 197 videos |
| Focused / Distracted | 2,626 / 649 (19.8% Distracted) |
| annotator agreement | Cohen's κ 0.696 to 0.715 on 514 windows labeled by all three |

Clips were deliberately enriched for off-task behavior, so 19.8% is not the natural rate in a
classroom.

This repository holds the code, window ids, labels and clip records. Videos, frames and skeleton
tensors are not included. Videos are available through the links published by the
[SAV authors](https://github.com/Ritatanz/SAV).

## Setup

Requires Python 3.10, `ffmpeg`, and an NVIDIA GPU.

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install scipy yt-dlp
.venv/bin/python -m pytest
```

For pretraining, download `ntu60_hrnet.pkl` as described in
[datasets/ntu/README.md](datasets/ntu/README.md).

## Running

```bash
# Data
.venv/bin/python steps/build_shortlist.py -n 200
.venv/bin/python steps/download_videos.py
.venv/bin/python steps/build_clip_list.py --high-frac 0.80
.venv/bin/python steps/cut_clips.py --clips datasets/sav/clip_list.txt
.venv/bin/python steps/extract_poses.py
.venv/bin/python steps/make_windows.py
.venv/bin/python steps/filter_windows.py

# Pretraining
.venv/bin/python steps/pretrain_ntu.py

# Experiments
bash steps/run_sweep.sh
bash steps/run_e_sweep.sh E1 E2 E3 E5 E6 E7

# Tables and reports, written to outputs/
.venv/bin/python steps/make_report.py
```

## Repository layout

```text
datasets/   labels, clip records and dataset notes
steps/      pipeline scripts, in the order above
lib/        models, data loading and shared code
analysis/   tables, statistics and figures from saved runs
tests/      test suite
outputs/    generated results (created when you run, not in git)
```
