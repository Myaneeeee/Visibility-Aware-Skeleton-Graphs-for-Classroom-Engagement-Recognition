"""Single source of truth for repository + data directory locations.

Every script imports these constants instead of deriving PROJECT_ROOT from its
own file location, so moving a script between folders never breaks data paths.

Layout::

    datasets/                 all data lives here
      sav/                    the working SAV dataset
        videos/               cut 3s clips
        pose_tracks/          *_timeline.jsonl    (extract_poses.py)
        windows/              *_windows.pt         (make_windows.py)
        windows_filtered/     *_windows.pt         (filter_windows.py) <- label + train on these
        clip_list.txt         clips to cut
        worklists/            per-annotator window lists (blind overlap)
        ledgers/              per-annotator label CSVs
        excluded_windows.csv  labeled windows kept out of training
      ntu/                    NTU-60 pretraining (ntu60_hrnet.pkl + exported backbone)
      annotations.csv         tracked ground-truth ledger (shared, committed)
    outputs/                  everything generated; safe to delete and regenerate
      README.md               written by steps/make_report.py -- start here
      reports/                human-readable markdown tables (the thesis numbers)
      figures/                .png figures, named <run>_<model>_<what>.png
      runs/<run>/<model>/     raw per-run artifacts (oof_probs.pt, metrics.json, loss.json)
      logs/                   stdout/stderr of sweep runs
"""
import os

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(LIB_DIR)

# --- top-level buckets ---
DATASETS_DIR = os.path.join(REPO_ROOT, "datasets")
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")

# --- SAV dataset (the working dataset) ---
SAV_DIR = os.path.join(DATASETS_DIR, "sav")
# Downloaded Youku source videos (~40 min each). Kept separate from videos/ so that
# extract_poses.py only ever sees the cut 3s clips -- pointing it at 40-minute
# sources would produce windows whose video_name is not a clip id.
SAV_VIDEOS_SOURCE_DIR = os.path.join(SAV_DIR, "videos_source")
SAV_DOWNLOAD_MANIFEST_FILE = os.path.join(SAV_VIDEOS_SOURCE_DIR, "download_manifest.csv")
SAV_VIDEOS_DIR = os.path.join(SAV_DIR, "videos")
SAV_POSE_TRACKS_DIR = os.path.join(SAV_DIR, "pose_tracks")
SAV_WINDOWS_DIR = os.path.join(SAV_DIR, "windows")
SAV_WINDOWS_FILTERED_DIR = os.path.join(SAV_DIR, "windows_filtered")
SAV_CLIP_LIST_FILE = os.path.join(SAV_DIR, "clip_list.txt")
SAV_WORKLIST_FILE = os.path.join(SAV_DIR, "annotation_worklist.txt")
SAV_WORKLISTS_DIR = os.path.join(SAV_DIR, "worklists")
SAV_LEDGERS_DIR = os.path.join(SAV_DIR, "ledgers")
SAV_ADJUDICATION_FILE = os.path.join(SAV_DIR, "adjudication_queue.csv")
SAV_ADJUDICATION_LABELS_FILE = os.path.join(SAV_DIR, "adjudication_labels.csv")
SAV_LABEL_PROVENANCE_FILE = os.path.join(SAV_DIR, "label_provenance.csv")
# Re-check of every window by reviewer (steps/review.py); one append-only CSV per reviewer.
SAV_REVIEW_DIR = os.path.join(SAV_DIR, "review")

# --- NTU-60 pretraining data ---
NTU_DIR = os.path.join(DATASETS_DIR, "ntu")
NTU_PKL_FILE = os.path.join(NTU_DIR, "ntu60_hrnet.pkl")
NTU_BACKBONE_FILE = os.path.join(NTU_DIR, "stgcn_ntu60_backbone.pth")

# --- shared label artifacts ---
ANNOTATIONS_FILE = os.path.join(DATASETS_DIR, "annotations.csv")
# Labeled windows kept out of training (window_id,reason).
EXCLUDED_WINDOWS_FILE = os.path.join(SAV_DIR, "excluded_windows.csv")

# --- outputs/ sub-buckets ---
# One bucket per audience: reports/ is what a human reads, figures/ is what goes
# in the thesis, runs/ is machine-readable input for both, logs/ is for debugging.
# Code that writes calls the functions below.

# --- per-run CV outputs ---
# Deliberately functions of a run label, not constants. The ablation ladder
# (PREREGISTRATION.md) runs four configs at several seeds each, and a fixed
# filename means every run overwrites the last -- so config B would destroy the
# out-of-fold probabilities config D needs from config A, with nothing on disk to
# show it happened. `run_label` comes from evaluate.run_label(config, seed).
#
# `model` is part of the path, not just the filename: every config trains an
# ST-GCN *and* the GRU/MLP baselines, and for a long time only the ST-GCN's
# predictions were kept -- which made it impossible to calibrate or plot the
# baseline that actually scores highest.

# Every helper below re-derives its bucket from OUTPUTS_DIR *at call time* rather
# than closing over the module-level constants. Tests redirect output by
# monkeypatching paths.OUTPUTS_DIR; if these read the constants instead, that
# redirect silently stops working and the suite writes into the real outputs/.

def runs_dir():
    return os.path.join(OUTPUTS_DIR, "runs")


def figures_dir():
    return os.path.join(OUTPUTS_DIR, "figures")


def reports_dir():
    return os.path.join(OUTPUTS_DIR, "reports")


def logs_dir():
    return os.path.join(OUTPUTS_DIR, "logs")


def run_dir(run_label, model="stgcn"):
    return os.path.join(runs_dir(), run_label, model)


def oof_probs_file(run_label, model="stgcn"):
    return os.path.join(run_dir(run_label, model), "oof_probs.pt")


def fold_metrics_file(run_label, model="stgcn"):
    return os.path.join(run_dir(run_label, model), "metrics.json")


def train_loss_file(run_label, model="stgcn"):
    return os.path.join(run_dir(run_label, model), "loss.json")


def cv_confusion_matrix_file(run_label, model="stgcn"):
    return os.path.join(figures_dir(), f"{run_label}_{model}_confusion.png")


def loss_curve_file(run_label, model="stgcn"):
    return os.path.join(figures_dir(), f"{run_label}_{model}_loss.png")


def diagnostics_dir():
    return os.path.join(OUTPUTS_DIR, "diagnostics")


def data_need_file(name):
    return os.path.join(diagnostics_dir(), name)


def representation_file(name="representation.json"):
    return os.path.join(diagnostics_dir(), name)


def figure_file(name):
    return os.path.join(figures_dir(), name)


def report_file(name):
    return os.path.join(reports_dir(), name)


def outputs_readme_file():
    return os.path.join(OUTPUTS_DIR, "README.md")

