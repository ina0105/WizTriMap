import os
import numpy as np
import torch
import trimap, umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100
from PIL import Image

# === Dataset Loader ===
def load_dataset(name, train=False):
    transform = transforms.ToTensor()
    if name == "MNIST":
        dataset = MNIST("./data", train=train, download=True, transform=transform)
        X = torch.stack([img[0].squeeze() for img in dataset])
    elif name == "FashionMNIST":
        dataset = FashionMNIST("./data", train=train, download=True, transform=transform)
        X = torch.stack([img[0].squeeze() for img in dataset])
    elif name == "CIFAR100":
        dataset = CIFAR100("./data", train=train, download=True, transform=transform)
        X = torch.stack([img[0] for img in dataset])
    else:
        raise ValueError("Unknown dataset")
    X_flat = X.view(X.size(0), -1).numpy()
    y = [label for _, label in dataset]
    return X, X_flat, y

def get_class_names(dataset_name):
    if dataset_name == "CIFAR100":
        return CIFAR100("./data", train=False).classes
    elif dataset_name == "FashionMNIST":
        return FashionMNIST("./data", train=False).classes
    elif dataset_name == "MNIST":
        return [str(i) for i in range(10)]
    else:
        return [str(i) for i in range(10)]

# ===Save Image ===
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
        im = Image.fromarray(img_uint8, mode="L")
    elif img_array.ndim == 3:
        img_uint8 = (img_array * 255).astype(np.uint8)
        im = Image.fromarray(img_uint8)
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")
    if resize_factor > 1:
        im = im.resize((im.width * resize_factor, im.height * resize_factor), interpolation)
    im.save(path)

# === Projection Functions ===
def project(method, X_flat, dim=2):
    if method == "TriMap":
        return trimap.TRIMAP(n_dims=dim).fit_transform(X_flat)
    elif method == "UMAP":
        return umap.UMAP(n_components=dim).fit_transform(X_flat)
    elif method == "t-SNE":
        return TSNE(n_components=dim).fit_transform(X_flat)
    elif method == "PCA":
        return PCA(n_components=dim).fit_transform(X_flat)
    else:
        raise ValueError("Unknown method")