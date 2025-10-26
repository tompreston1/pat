import os
import csv
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt

from modules import ConvNeXtBinary
from dataset import ADNIDataset, scan_folder, get_transforms


# ------------------------------
# Utility: Reproducibility
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------
# Train / Eval helpers
# ------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    scaler=None, accumulation_steps=1):
    model.train()
    losses, y_true, y_pred = [], [], []
    step = 0

    for X, y in tqdm(loader, desc="Train"):
        X = X.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16) if scaler is not None else torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(X)
            loss = criterion(logits, y) / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # OneCycleLR expects per-iteration stepping
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item() * accumulation_steps)
        preds = logits.argmax(1).detach().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.detach().cpu().numpy())

        step += 1

    acc = accuracy_score(y_true, y_pred)
    return np.mean(losses), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Val"):
    model.eval()
    losses, y_true, y_pred = [], [], []
    y_scores = []  # probabilities for ROC-AUC

    for X, y in tqdm(loader, desc=desc):
        X = X.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)
        losses.append(loss.item())

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_scores.extend(probs)
        preds = logits.argmax(1).detach().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.detach().cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = float("nan")
    return np.mean(losses), acc, f1, auc, (np.array(y_true), np.array(y_pred))


def save_curves(epochs, train_accs, val_accs, train_losses, val_losses, out_png="training_curves.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"📈 Saved curves to {out_png}")


def save_confusion_matrix(y_true, y_pred, out_png="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'],
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')

    # annotate counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"🧮 Saved confusion matrix to {out_png}")


# ------------------------------
# Main training logic
# ------------------------------
def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}")

    # --- Load train/test folders ---
    train_dir = os.path.join(args.data_root, "train")
    test_dir = os.path.join(args.data_root, "test")

    train_samples = scan_folder(train_dir)
    test_samples = scan_folder(test_dir)

    if len(train_samples) == 0:
        raise ValueError(f"No training images found under {train_dir}")
    if len(test_samples) == 0:
        raise ValueError(f"No testing images found under {test_dir}")

    # --- Split train/val ---
    random.shuffle(train_samples)
    n_train = int(len(train_samples) * 0.85)
    val_samples = train_samples[n_train:]
    train_samples = train_samples[:n_train]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # --- Check class balance ---
    tr_cnt = Counter(l for _, l in train_samples)
    va_cnt = Counter(l for _, l in val_samples)
    te_cnt = Counter(l for _, l in test_samples)
    print("Train class distribution:", tr_cnt)
    print("Val class distribution:", va_cnt)
    print("Test class distribution:", te_cnt)

    # --- Datasets & DataLoaders ---
    transform_train = get_transforms(train=True)
    transform_eval = get_transforms(train=False)

    train_ds = ADNIDataset(train_samples, transform=transform_train)
    val_ds = ADNIDataset(val_samples, transform=transform_eval)
    test_ds = ADNIDataset(test_samples, transform=transform_eval)

    # Optional: use class weights in the loss (balanced loss)
    class_weights = None
    if args.balanced_loss:
        total = tr_cnt[0] + tr_cnt[1]
        w0 = total / (2.0 * tr_cnt[0])
        w1 = total / (2.0 * tr_cnt[1])
        class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        print(f"Using class weights: {w0:.4f} (NC), {w1:.4f} (AD)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # --- Model, loss, optimizer, scheduler ---
    model = ConvNeXtBinary(model_name=args.model_name, pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # One-Cycle LR with warmup (step each batch)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.warmup_pct,     # warmup portion of total steps
        div_factor=25.0,               # initial lr = max_lr / div_factor
        final_div_factor=1e4,          # final lr = initial lr / final_div_factor
        anneal_strategy='cos'
    )

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- Metrics tracking & logging ---
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    val_f1s, val_aucs = [], []
    best_val_acc = 0.0
    epochs_no_improve = 0

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "metrics_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_f1", "val_auc"])

    # --- Training loop ---
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            scaler=scaler, accumulation_steps=args.accumulation_steps
        )
        val_loss, val_acc, val_f1, val_auc, _ = evaluate(model, val_loader, criterion, device, desc="Val")

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)

        print(f"[Epoch {epoch}] "
              f"train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}, "
              f"val_f1={val_f1:.3f}, val_auc={val_auc:.3f}")

        # CSV log
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, val_loss, tr_acc, val_acc, val_f1, val_auc])

        # Early stopping on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, args.ckpt))
            print("✅ Saved best model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"⏹️ Early stopping at epoch {epoch} (no val_acc improvement for {args.early_stop_patience} epochs).")
                break

    # --- Final test evaluation (load best) ---
    best_ckpt_path = os.path.join(args.out_dir, args.ckpt)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_loss, test_acc, test_f1, test_auc, (y_true, y_pred) = evaluate(model, test_loader, criterion, device, desc="Test")
    print(f"\nFinal Test — Loss={test_loss:.4f}, Acc={test_acc:.3f}, F1={test_f1:.3f}, AUC={test_auc:.3f}")
    print("✅ Training complete.")

    # --- Save plots ---
    epochs_axis = list(range(len(train_accs)))
    save_curves(epochs_axis, train_accs, val_accs, train_losses, val_losses,
                out_png=os.path.join(args.out_dir, "training_curves.png"))
    save_confusion_matrix(y_true, y_pred, out_png=os.path.join(args.out_dir, "confusion_matrix.png"))


# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='Path to AD/NC folders (containing train/ and test/)')
    parser.add_argument('--model_name', default='convnext_tiny')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--warmup_pct', type=float, default=0.1, help='OneCycle warmup percent of total steps')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--balanced_loss', action='store_true', help='Use class-weighted CrossEntropy')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ckpt', default='best_model.pth')
    parser.add_argument('--out_dir', default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
