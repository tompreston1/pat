# train.py
import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from modules import ConvNeXtBinary
from dataset import ADNIDataset
from torchvision import transforms

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def scan_folder(base_dir):
    """Scans AD/NC subfolders for .jpg/.jpeg/.png files and labels them."""
    samples = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, f)
                if 'ad' in root.lower():
                    label = 1
                elif 'nc' in root.lower() or 'cn' in root.lower():
                    label = 0
                else:
                    continue
                samples.append((path, label))
    return samples

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, y_true, y_pred = [], [], []
    for X, y in tqdm(loader, desc='Train'):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(1).detach().cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    return np.mean(losses), acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc='Val'):
    model.eval()
    losses, y_true, y_pred = [], [], []
    for X, y in tqdm(loader, desc=desc):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = logits.argmax(1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    return np.mean(losses), acc

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load all samples
    all_samples = scan_folder(args.data_root)
    if len(all_samples) == 0:
        raise ValueError(f"No image files found under {args.data_root}")

    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    test_samples = all_samples[n_train + n_val:]

    print(f"Total: {n}, Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_ds = ADNIDataset(train_samples, transform=transform_train)
    val_ds   = ADNIDataset(val_samples,   transform=transform_eval)
    test_ds  = ADNIDataset(test_samples,  transform=transform_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ConvNeXtBinary(model_name=args.model_name, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_val_acc = 0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] train_acc={tr_acc:.3f}, val_acc={val_acc:.3f}")
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.ckpt)
            print("✅ Saved best model")

    # Final evaluation
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, desc='Test')
    print(f"\nFinal Test Results — Acc={test_acc:.3f}")
    print("✅ Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True, help='Path to AD/NC folders')
    parser.add_argument('--model_name', default='convnext_tiny')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ckpt', default='best_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
