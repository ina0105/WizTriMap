import os
import argparse
import numpy as np
from pathlib import Path
from utils import load_dataset, project

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Save projection embeddings.")
parser.add_argument("--dim", type=int, default=2, help="Target dimension for TriMap")
parser.add_argument("--out-dir", type=str, default="precomputed_embeddings", help="Directory to save .npy files")
parser.add_argument("--force", action="store_true", help="Force overwrite of existing .npy files")
args = parser.parse_args()

# -----------------------------
# Projection methods
# -----------------------------
projection_methods = ["TriMap", "UMAP", "t-SNE", "PCA"]
datasets = ["MNIST", "FashionMNIST", "CIFAR100"]

# -----------------------------
# Save
# -----------------------------
os.makedirs(args.out_dir, exist_ok=True)

for dataset in datasets:
    print(f"\n[INFO] Dataset: {dataset}")
    _, X_flat, _ = load_dataset(dataset)

    for method in projection_methods:
        if method != "TriMap":
            if args.dim != 2:
                print(f"[SKIP] {method} only supports 2D. Skipping dim={args.dim}.")
                continue
            filename = f"{dataset}_{method}_2D.npy"
            dim = 2
        else:
            filename = f"{dataset}_{method}_{args.dim}D.npy"
            dim = args.dim

        save_path = Path(args.out_dir) / filename

        if save_path.exists() and not args.force:
            print(f"[SKIP] {filename} already exists. Use --force to overwrite.")
            continue

        print(f"[PROJECT] {method} with dim={dim}")
        try:
            emb = project(method, X_flat, dim)
            np.save(save_path, emb)
            print(f"[SAVED] {filename}")
        except Exception as e:
            print(f"[ERROR] Failed for {method} on {dataset}: {e}")
