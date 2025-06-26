import os
import sys
import json
import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers.config import dataset_list, method_list
from models.models import ConvDecoderGray, ConvDecoder, combined_ssim_mse_loss
from utils.save_projections_utils import load_dataset, save_image, project

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(x_tensor, embedding, model, loss_fn, save_path, epochs, method, dataset, dim,
                learning_rate=0.0001, train_batch_size=64, val_batch_size=64, visualize=False):
    # Convert full tensors
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
    image_tensor = x_tensor.cpu()   # Keep on CPU until batching to GPU

    indices = np.arange(len(embedding_tensor))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    train_ds = TensorDataset(embedding_tensor[train_idx], image_tensor[train_idx])
    val_ds = TensorDataset(embedding_tensor[val_idx], image_tensor[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=False)

    best_val_loss = float("inf")
    best_state_dict = None

    vis_dir = os.path.join("recon_viz", dataset.replace("-", "_"), method)
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)

    for epoch in tqdm(range(epochs), desc=f"Training {os.path.basename(save_path)}"):
        model.train()
        train_loss = 0
        for emb_batch, img_batch in train_loader:
            emb_batch = emb_batch.to(device)
            img_batch = img_batch.to(device)
            optimizer.zero_grad()
            out = model(emb_batch)
            loss = loss_fn(out, img_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img_batch.size(0)

        model.eval()
        val_loss_total, correct, total = 0, 0, 0
        with torch.no_grad():
            for emb_batch, img_batch in val_loader:
                emb_batch = emb_batch.to(device)
                img_batch = img_batch.to(device)
                out = model(emb_batch)
                val_loss = loss_fn(out, img_batch)
                val_loss_total += val_loss.item() * img_batch.size(0)

                pred = (out > 0.5)
                true = (img_batch > 0.5)
                correct += (pred == true).float().sum().item()
                total += img_batch.numel()

        avg_val_loss = val_loss_total / len(val_ds)
        val_acc = correct / total
        tqdm.write(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss: .4f} | Val Acc: {val_acc: .4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()

        # Visualization every 100 epochs
        if visualize and (epoch + 1) % 100 == 0:
            # === Reconstructed Image ===
            sample_latent = embedding_tensor[val_idx[10]:val_idx[10]+1].to(device)
            with torch.no_grad():
                recon_img = model(sample_latent).cpu().numpy()

            if recon_img.shape[1] == 1:
                img = recon_img[0, 0]
            else:
                img = recon_img[0].transpose(1, 2, 0)

            recon_vis = (img - img.min()) / (img.max() - img.min() + 1e-5)
            save_image(recon_vis, os.path.join(vis_dir,
                                               f"{dataset.replace('-', '_')}_{method}_{dim}D_epoch{epoch+1}_recon.png"),
                       resize_factor=8)

            original = image_tensor[val_idx[0]].numpy()
            if original.shape[0] == 1:
                original_img = original[0]
            else:
                original_img = original.transpose(1, 2, 0)
            save_image(original_img, os.path.join(vis_dir, f"{dataset.replace('-', '_')}_{method}_{dim}D_epoch{epoch+1}_original.png"),
                       resize_factor=8)

    # Final eval on val set using best model
    model.load_state_dict(best_state_dict)
    model.eval()
    val_recon = []
    val_targets = []
    with torch.no_grad():
        for emb_batch, img_batch in val_loader:
            emb_batch = emb_batch.to(device)
            out = model(emb_batch).cpu()
            val_recon.append(out)
            val_targets.append(img_batch)

    recon_all = torch.cat(val_recon, dim=0)
    target_all = torch.cat(val_targets, dim=0)
    recon_error = F.mse_loss(recon_all, target_all).item()

    torch.save(best_state_dict, save_path)
    print(f"[SAVED BEST] {save_path} | Best Val Loss: {best_val_loss: .4f} | Recon MSE: {recon_error: .4f}")
    return recon_error


def main(args):
    os.makedirs(os.path.join("..", args.save_dir), exist_ok=True)
    recon_errors = {}

    trimap_dims = list(range(2, args.ndims+1))
    root_dir = os.path.join("..", args.save_dir)

    for dataset in dataset_list:
        print(f"\nLoading full dataset: {dataset}")
        x, x_flat, _ = load_dataset(dataset, train=True)
        x_tensor = torch.tensor(x_flat, dtype=torch.float32).to(device)

        for method in method_list:
            dims = trimap_dims if method == "TriMap" else [2]
            for dim in dims:
                print(f"Projecting {dataset} using {method} ({dim}D)")
                emb = project(method, x_flat, dim)

                save_path = os.path.join(root_dir, f"{dataset.replace('-', '_')}_{method}_{dim}D.pth")

                if dataset == "CIFAR-100":
                    model = ConvDecoder(input_dim=dim).to(device)
                    x_img = x_tensor.view(-1, 3, 32, 32)
                    loss_fn = lambda o, t: combined_ssim_mse_loss(o, t, alpha=0.3)
                    epochs = args.epochs_cifar
                else:
                    model = ConvDecoderGray(input_dim=dim).to(device)
                    x_img = x_tensor.view(-1, 1, 28, 28)
                    loss_fn = lambda o, t: combined_ssim_mse_loss(o, t, alpha=0.5)
                    epochs = args.epochs_mnist

                recon_mse = train_model(x_img, emb, model, loss_fn, save_path, epochs, method,
                                        dataset, dim, learning_rate=args.lr, train_batch_size=args.train_batchsize, 
                                        val_batch_size=args.val_batchsize, visualize=args.visualize)

                key = f'{dataset.replace("-", "_")}_{method}_{dim}D'
                recon_errors[key] = recon_mse

    with open(os.path.join(root_dir, "recon_errors.json"), "w") as f:
        json.dump(recon_errors, f, indent=2)

    print("[SAVED] Reconstruction error summary at recon_errors.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save inversion models for DR embeddings")
    parser.add_argument("--epochs_cifar", type=int, default=500, help="Epochs for CIFAR100 models")
    parser.add_argument("--epochs_mnist", type=int, default=300, help="Epochs for MNIST and FashionMNIST models")
    parser.add_argument("--ndims", type=int, default=7,
                        help="Number of dimensions in power of 2 to reduce to for TriMap")
    parser.add_argument("--lr", type=int, default=0.0001, help="Learning rate")
    parser.add_argument("--train_batchsize", type=int, default=64, help="Training Batch size")
    parser.add_argument("--val_batchsize", type=int, default=64, help="Validation Batch size")
    parser.add_argument("--save_dir", type=str, default="multi_dimension_inversion_models",
                        help="Directory to save trained models")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization every 100 epochs (TriMap only)")

    arguments = parser.parse_args()
    main(arguments)
