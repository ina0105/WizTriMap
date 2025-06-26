import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.save_projections_utils import load_dataset, project
from helpers.config import dataset_list, method_list


def save_projections(args):
    os.makedirs(args.out_dir, exist_ok=True)

    for dataset in dataset_list:
        print(f"\n[INFO] Dataset: {dataset}")
        _, x_flat, _ = load_dataset(dataset)

        for method in method_list:
            if method != "TriMap":
                if args.dim != 2:
                    print(f"[SKIP] {method} only supports 2D. Skipping dim={args.dim}.")
                    continue
                filename = f"{dataset}_{method}_2D.npy"
                dim = 2
            else:
                filename = f"{dataset}_{method}_{args.dim}D.npy"
                dim = args.dim

            save_path = os.path.join(args.out_dir, filename)

            if os.path.exists(save_path) and not args.force:
                print(f"[SKIP] {filename} already exists. Use --force to overwrite.")
                continue

            print(f"[PROJECT] {method} with dim={dim}")
            try:
                emb = project(method, x_flat, dim)
                np.save(save_path, emb)
                print(f"[SAVED] {filename}")
            except Exception as e:
                print(f"[ERROR] Failed for {method} on {dataset}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save projection embeddings.")
    parser.add_argument("--dim", type=int, default=2, help="Target dimension for TriMap")
    parser.add_argument("--out_dir", type=str, default="precomputed_embeddings", help="Directory to save .npy files")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing .npy files")
    arguments = parser.parse_args()

    save_projections(arguments)
