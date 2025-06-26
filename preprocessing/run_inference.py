import os
import sys
import argparse
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import ConvDecoderGray, ConvDecoder
from utils.save_projections_utils import load_dataset, save_image, project, get_class_names
from helpers.config import dataset_list, method_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(dataset, method, dim, save_dir):
    path = os.path.join('..', save_dir, f"{dataset}_{method}_{dim}D.pth")
    if dataset == "CIFAR-100":
        model = ConvDecoder(input_dim=dim).to(device)
    else:
        model = ConvDecoderGray(input_dim=dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def run_all(args):
    for dataset in dataset_list:
        x, x_flat, y_true = load_dataset(dataset)
        class_names = get_class_names(dataset)

        for method in method_list:
            print(f"\n>>> Running {method} on {dataset} with dim={args.dim}")
            model = load_model(dataset, method, args.dim, args.save_dir)

            embedding = project(method, x_flat, args.dim)
            emb_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

            batch_size = 256
            recon_errors = np.zeros(len(x))
            save_indices = set(random.sample(range(len(x)), 20))

            # Output folder
            save_path = os.path.join(args.output_dir, dataset, f"{method}_{args.dim}D")
            os.makedirs(save_path, exist_ok=True)

            with torch.no_grad():
                for i in tqdm(range(0, len(emb_tensor), batch_size)):
                    batch = emb_tensor[i: i + batch_size]
                    output = model(batch).cpu()

                    if dataset == "CIFAR-100":
                        target = x[i: i + batch_size].view(-1, 3, 32, 32)
                    else:
                        target = x[i: i + batch_size].view(-1, 1, 28, 28)

                    recon = output 

                    batch_errors = F.mse_loss(recon, target, reduction="none").mean(dim=(1, 2, 3)).numpy()
                    recon_errors[i: i + batch_size] = batch_errors

                    for j in range(recon.size(0)):
                        global_idx = i + j
                        if global_idx not in save_indices:
                            continue
                        recon_img = recon[j]
                        orig_img = target[j]
                        class_name = class_names[y_true[global_idx]]
                        safe_class_name = class_name.replace("/", "_").replace(" ", "_")
                        if dataset == "CIFAR-100":
                            img = recon_img.permute(1, 2, 0).numpy()
                            orig_np = orig_img.permute(1, 2, 0).numpy()
                        else:
                            img = recon_img[0].numpy()
                            orig_np = orig_img[0].numpy()

                        # Save reconstructed image
                        filename = f"recon_{global_idx: 04d}_true_{safe_class_name}.png"
                        save_image(img, os.path.join(save_path, filename), resize_factor=4)

                        # Save original image
                        orig_filename = f"orig_{global_idx: 04d}_true_{safe_class_name}.png"
                        save_image(orig_np, os.path.join(save_path, orig_filename), resize_factor=4)

            # Save recon error
            np.save(os.path.join(save_path, "recon_errors.npy"), recon_errors)
            print(f"[DONE] {dataset}-{method}: Saved recon errors + 20 samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, required=True, help="Embedding dimensionality (e.g., 2, 3, ..., 7)")
    parser.add_argument("--save_dir", type=str, default="multi_dimension_inversion_models",
                        help="Path to saved .pth models")
    parser.add_argument("--output_dir", type=str, default="recon_output",
                        help="Where to save reconstructed images and errors")
    arguments = parser.parse_args()
    run_all(arguments)
