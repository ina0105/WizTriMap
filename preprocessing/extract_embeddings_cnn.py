import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers.config import dataset_list, method_list
from trimap.trimap import TRIMAP


class CNN5Layer(nn.Module):
    def __init__(self, num_output_classes=10, dimension=1):
        super(CNN5Layer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(dimension, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_output_classes)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_pool = self.global_pool(x3)
        x_flat = x_pool.view(x_pool.size(0), -1)
        out = self.classifier(x_flat)
        return out, [x1, x2, x3]


def get_dataloader(dataset_to_load, batch_size=64, imagenet_subset=10000):
    if dataset_to_load == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    elif dataset_to_load == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    elif dataset_to_load == 'CIFAR-100':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize to match MNIST dimensions
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
        ])
        dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset')
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
                      persistent_workers=True)


def train_model(model, data_loader, device, epochs=1):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_images, batch_labels in data_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    print('Training complete.')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join('..', 'cnn_layer_embeddings_trimap'), exist_ok=True)
    num_classes = 10
    dim = 1
    for dataset_name in dataset_list:
        print(f'Processing {dataset_name}...')
        if dataset_name == 'MNIST':
            dim = 1
            num_classes = 10
        elif dataset_name == 'FashionMNIST':
            dim = 1
            num_classes = 10
        elif dataset_name == 'CIFAR-100':
            dim = 3
            num_classes = 100

        dataloader = get_dataloader(dataset_name, batch_size=128)
        cnn_model = CNN5Layer(num_output_classes=num_classes, dimension=dim).to(device)
        for images, labels in dataloader:
            print(images.shape)
        train_model(cnn_model, dataloader, device, epochs=5)

        cnn_model.eval()
        all_embeddings = [[] for _ in range(3)]
        all_labels = []
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                _, layer_outputs = cnn_model(images)
                for i, layer_out in enumerate(layer_outputs):
                    batch_flat = layer_out.view(layer_out.size(0), -1).cpu()
                    all_embeddings[i].append(batch_flat)
                all_labels.append(labels.cpu())

        # Concatenate all batches
        for i in range(3):
            all_embeddings[i] = torch.cat(all_embeddings[i], dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        figures = {}
        # Check if embeddings already exist
        for method in method_list:
            for layer in range(1, 4):
                file_path = os.path.join("..", 'cnn_layer_embeddings_trimap',
                                         f'{dataset_name.replace("-", "_")}_{method}_layer{layer}_embeddings.pt')

                # Perform dimensionality reduction
                if method == 'PCA':
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=2)
                elif method == 't_SNE':
                    from sklearn.manifold import TSNE
                    reducer = TSNE(n_components=2, random_state=42)
                elif method == 'UMAP':
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                elif method == 'Trimap':
                    reducer = TRIMAP()
                else:
                    raise ValueError(f'Unknown method: {method}')

                reduced_embeddings = reducer.fit_transform(np.array(all_embeddings[layer]))
                torch.save(reduced_embeddings, file_path)
        print(f'Embeddings and labels saved for {dataset_name} in cnn_layer_embeddings_trimap/')
