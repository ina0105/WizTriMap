import os
import sys
import json
import argparse
import torch
import numpy as np
import umap
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from trimap.trimap import TRIMAP

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import ConvDecoder, ConvDecoderGray, combined_ssim_mse_loss
from helpers.config import dataset_list,method_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_image(img_array, path, resize_factor=4, interpolation=Image.NEAREST):
    """
    Save grayscale or RGB image with optional resizing.

    Args:
        img_array: numpy array in [0,1], shape (H,W) or (H,W,3)
        path: output file path
        resize_factor: scale up by this factor (e.g., 4 x 32 = 128 x 128)
        interpolation: PIL.Image resize mode (e.g., NEAREST or BILINEAR)
    """
    img_array = np.clip(img_array, 0, 1)
    
    if img_array.ndim == 2:
        img_uint8 = (img_array * 255).astype(np.uint8)
        im = Image.fromarray(img_uint8, mode='L')
    elif img_array.ndim == 3:
        img_uint8 = (img_array * 255).astype(np.uint8)
        im = Image.fromarray(img_uint8)
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")

    if resize_factor > 1:
        new_size = (im.width * resize_factor, im.height * resize_factor)
        im = im.resize(new_size, interpolation)

    im.save(path)


def project(name, method, x_flat, dim=2):
    if method == "TriMap":
        return TRIMAP(n_dims=dim).fit_transform(x_flat)
    elif method == "UMAP":
        return umap.UMAP(n_components=dim).fit_transform(x_flat)
    elif method == "t-SNE":
        return TSNE(n_components=dim).fit_transform(x_flat)
    elif method == "PCA":
        return PCA(n_components=dim).fit_transform(x_flat)
    else:
        raise ValueError("Unknown method")


def load_dataset(name):
    transform = transforms.ToTensor()
    if name == "MNIST":
        dataset = MNIST("../data", train=True, download=True, transform=transform)
        x = torch.stack([img[0].squeeze() for img in dataset])
    elif name == "FashionMNIST":
        dataset = FashionMNIST("../data", train=True, download=True, transform=transform)
        x = torch.stack([img[0].squeeze() for img in dataset])
    elif name == "CIFAR-100":
        dataset = CIFAR100("../data", train=True, download=True, transform=transform)
        x = torch.stack([img[0] for img in dataset])
    else:
        raise ValueError("Invalid dataset")

    x_flat = x.view(x.size(0), -1).numpy()
    return x, x_flat


def train_model(x_tensor, embedding, model, loss_fn, save_path, epochs, method, dataset, dim, visualize=False):
    # Convert full tensors
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
    
    # Make sure image tensor is on CPU initially
    if isinstance(x_tensor, torch.Tensor):
        image_tensor = x_tensor.cpu()
    else:
        image_tensor = torch.tensor(x_tensor, dtype=torch.float32)
        
    # Check that shapes are compatible
    print(f"Data shapes - Images: {image_tensor.shape}, Embeddings: {embedding_tensor.shape}")

    # Check that embedding and image counts match
    if len(embedding_tensor) != len(image_tensor):
        print(f"WARNING: Mismatch in counts - Embeddings: {len(embedding_tensor)}, Images: {len(image_tensor)}")
        # Use the smaller of the two sizes
        min_size = min(len(embedding_tensor), len(image_tensor))
        embedding_tensor = embedding_tensor[:min_size]
        image_tensor = image_tensor[:min_size]
        print(f"Truncated to {min_size} samples")
    
    indices = np.arange(len(embedding_tensor))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    train_ds = TensorDataset(embedding_tensor[train_idx], image_tensor[train_idx])
    val_ds = TensorDataset(embedding_tensor[val_idx], image_tensor[val_idx])
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30)

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
        tqdm.write(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()

        # Visualization every 10 epochs for TriMap
        if visualize and (epoch + 1) % 100 == 0:
            # === Reconstructed Image ===
            sample_latent = embedding_tensor[val_idx[0]:val_idx[0]+1].to(device)
            with torch.no_grad():
                recon_img = model(sample_latent).cpu().numpy()

            if recon_img.shape[1] == 1:
                img = recon_img[0, 0]  # (1, H, W) -> (H, W)
            else:
                img = recon_img[0].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)

            recon_vis = (img - img.min()) / (img.max() - img.min() + 1e-5)

            save_image(recon_vis, os.path.join(vis_dir,
                                               f"{dataset.replace('-', '_')}_{method}_{dim}D_epoch{epoch+1}_recon.png"),
                       resize_factor=8)

            # === Original Image ===
            original = image_tensor[val_idx[0]].numpy()
            if original.shape[0] == 1:
                original_img = original[0]
            else:
                original_img = original.transpose(1, 2, 0)

            save_image(original_img, os.path.join(vis_dir,
                                                  f"{dataset}_{method}_{dim}D_epoch{epoch+1}_original.png"),
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


def load_emmbeddings(dataset, method, layer):
    """
    Load precomputed embeddings for each dataset.
    Returns a dictionary with dataset names as keys and embeddings as values.
    """
    embeddings_dir = os.path.join('..', 'cnn_layer_embeddings_trimap')
    file_path = os.path.join(embeddings_dir, f"{dataset.replace('-', '_')}_{method}_layer{layer}_embeddings.pt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings file {file_path} does not exist. Please run the embedding extraction script first.")

    embeddings = torch.load(file_path, map_location=device)
    return embeddings


def main(args):
    os.makedirs(os.path.join('..', args.save_dir), exist_ok=True)
    recon_errors = {}

    for dataset in dataset_list:
        print(f"\nLoading full dataset: {dataset}")
        x, x_flat = load_dataset(dataset)
        for method in method_list:
            for layer in range(1, 4):
                model_path = os.path.join(args.save_dir, f"{dataset.replace('-', '_')}_{method}_{layer}layer.pth")
                if os.path.exists(model_path):
                    print(f"[SKIP] Model {model_path} already exists. Skipping...")
                    continue
                print(f"Projecting {dataset} using {method} (layer {layer})")

                emb = load_emmbeddings(dataset=dataset, method=method, layer=layer)
                dim = 2 
                save_path = os.path.join(args.save_dir, f"{dataset.replace('-', '_')}_{method}_layer{layer}.pth")

                if dataset == "CIFAR-100":
                    if x.shape[1] == 1:
                        print("Loading CIFAR data for reconstruction...")
                        cifar_x, _ = load_dataset("CIFAR-100")
                        x_img = cifar_x.to(device)
                    else:
                        x_img = x.view(-1, 3, 32, 32).to(device)
                    
                    model = ConvDecoder(input_dim=dim).to(device)
                    loss_fn = lambda o, t: combined_ssim_mse_loss(o, t, alpha=0.3)
                    epochs = args.epochs_cifar
                else:
                    if x.shape[1] == 3:
                        print("Loading MNIST/Fashion data for reconstruction...")
                        if dataset == "mnist":
                            mnist_x, _ = load_dataset("MNIST")
                            x_img = mnist_x.view(-1, 1, 28, 28).to(device)
                        else:
                            fashion_x, _ = load_dataset("FashionMNIST")
                            x_img = fashion_x.view(-1, 1, 28, 28).to(device)
                    else:
                        x_img = x.view(-1, 1, 28, 28).to(device)
                    
                    model = ConvDecoderGray(input_dim=dim).to(device)
                    loss_fn = lambda o, t: combined_ssim_mse_loss(o, t, alpha=0.5)
                    epochs = args.epochs_mnist

                recon_mse = train_model(x_img, emb, model, loss_fn, save_path, epochs, method, dataset, dim,
                                        visualize=args.visualize)

                key = f"{dataset.replace('-', '_')}_{method}_layer{layer}"
                recon_errors[key] = recon_mse

    with open(os.path.join('..', args.save_dir, "recon_errors.json"), "w") as f:
        json.dump(recon_errors, f, indent=2)

    print("[SAVED] Reconstruction error summary at recon_errors.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save inversion models for DR embeddings")
    parser.add_argument("--epochs_cifar", type=int, default=1000, help="Epochs for CIFAR100 models")
    parser.add_argument("--epochs_mnist", type=int, default=500, help="Epochs for MNIST and FashionMNIST models")
    parser.add_argument("--ndims", type=int, default=10, help="Max number of dimensions to reduce to for TriMap")
    parser.add_argument("--save_dir", type=str, default="cnn_layer_inversion_models",
                        help="Directory to save trained models")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization every 10 epochs (TriMap only)")

    arguments = parser.parse_args()

    main(arguments)
