import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
from model import STGCNClassifier
from ntu_dataset import NTUDataset, load_split, subject_holdout
from paths import NTU_PKL_FILE, NTU_BACKBONE_FILE


def export_backbone(state_dict):
    """Return a copy of state_dict with the classification head (fc.*) removed."""
    return OrderedDict((k, v) for k, v in state_dict.items()
                       if not k.startswith("fc."))


def _run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item())
            correct += int((out.argmax(1) == y).sum())
            seen += y.numel()
    return total / max(1, len(loader)), correct / max(1, seen)   # (loss, top-1 acc)


def train_backbone(train_ds, val_ds, *, epochs=30, lr=1e-3, batch_size=64,
                   patience=5, device="cpu", num_classes=60,
                   out_path=NTU_BACKBONE_FILE):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if len(val_ds) == 0:
        raise ValueError(
            "Empty validation set: subject_holdout produced no val subjects. "
            "Increase --limit to span >=2 subjects, or raise --val-frac."
        )
    model = STGCNClassifier(in_channels=3, hidden_dim=128,
                            num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()      # NTU60 ~ class-balanced; plain CE

    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                    drop_last=len(train_ds) > batch_size)
    va = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history, best_val, no_improve = [], float("inf"), 0
    best_state, best_train_acc = export_backbone(model.state_dict()), 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, tr, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, va, criterion, device)
        history.append({"epoch": epoch, "train": tr_loss, "val": va_loss,
                        "train_acc": tr_acc, "val_acc": va_acc})
        print(f"epoch {epoch:03d} | train {tr_loss:.4f} (acc {tr_acc:.3f}) | "
              f"val {va_loss:.4f} (acc {va_acc:.3f})")
        if va_loss < best_val:
            best_val, best_train_acc, no_improve = va_loss, tr_acc, 0
            best_state = export_backbone(model.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stop at epoch {epoch}")
                break

    torch.save(best_state, out_path)
    # Honest sanity number (NOT a SOTA claim): NTU60 top-1 the backbone reached.
    print(f"saved backbone -> {out_path} | NTU60 train top-1 @ best val: {best_train_acc:.3f}")
    return {"history": history, "best_val": best_val,
            "best_train_acc": best_train_acc, "ckpt_path": out_path}


def main():
    ap = argparse.ArgumentParser(description="Pretrain ST-GCN backbone on NTU60-2D")
    ap.add_argument("--pkl", default=NTU_PKL_FILE, help="path to ntu60_hrnet.pkl")
    ap.add_argument("--split", default="xsub_train")
    ap.add_argument("--out", default=NTU_BACKBONE_FILE)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    annos = load_split(args.pkl, split=args.split, limit=args.limit)
    tr_annos, va_annos = subject_holdout(annos, val_frac=args.val_frac)
    print(f"{len(annos)} samples | train {len(tr_annos)} | val {len(va_annos)}")
    train_backbone(NTUDataset(tr_annos), NTUDataset(va_annos),
                   epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                   device=args.device, out_path=args.out)


if __name__ == "__main__":
    main()
