# predict.py
import argparse
import torch
from modules import ConvNeXtBinary
from dataset import ADNIDataset, make_transforms
from torch.utils.data import DataLoader

def load_csv(csv):
    import pandas as pd
    df = pd.read_csv(csv)
    return [(r['filepath'], int(r['label'])) for _, r in df.iterrows()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNeXtBinary()
    model.to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck['state_dict'])
    model.eval()

    samples = load_csv(args.csv)
    ds = ADNIDataset(samples, transform=make_transforms('val'))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    import numpy as np
    all_probs = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            all_probs.extend(probs.tolist())
    print("Predicted probabilities (first 20):", all_probs[:20])
