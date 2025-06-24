import random
import numpy as np
import torch
from torchvision.datasets import MNIST, FashionMNIST, CIFAR100
from torchvision import transforms
from torch.utils.data import Subset


class MNIST_Dataset:
    data = None
    data_numpy = None
    data_indices = None
    data_labels = None
    classes = None
    class_ids = None
    original_subset_indices = None
    current_subset_indices = None
    current_subset = None
    current_subset_numpy = None
    current_labels = None

    @staticmethod
    def load_data(downloaded_data):
        MNIST_Dataset.data = downloaded_data
        data_images = torch.stack([img[0].squeeze() for img in downloaded_data])
        data_images_flat = data_images.view(data_images.size(0), -1).numpy()
        MNIST_Dataset.data_numpy = data_images_flat
        MNIST_Dataset.data_indices = np.array(list(range(len(downloaded_data))))
        MNIST_Dataset.classes = [x.split('-')[-1].strip() for x in downloaded_data.classes]
        data_labels = [MNIST_Dataset.classes[downloaded_data[i][1]] for i in range(len(downloaded_data))]
        MNIST_Dataset.data_labels = data_labels
        MNIST_Dataset.class_ids = [int(x.split('-')[0].strip()) for x in downloaded_data.classes]

    @staticmethod
    def get_data_subset(downloaded_data, sample_size=5000):
        sample_indices = random.sample(range(len(downloaded_data)), sample_size)
        MNIST_Dataset.original_subset_indices = sample_indices
        MNIST_Dataset.current_subset_indices = np.array(list(range(sample_size)))
        subset = Subset(downloaded_data, sample_indices)
        MNIST_Dataset.current_subset = subset
        x = torch.stack([img[0].squeeze() for img in subset])
        x_flat = x.view(x.size(0), -1).numpy()
        labels = [MNIST_Dataset.classes[downloaded_data[i][1]] for i in sample_indices]
        return x_flat, labels

    @staticmethod
    def load_current_subset(data_subset_numpy, data_subset_labels):
        MNIST_Dataset.current_subset_numpy = data_subset_numpy
        MNIST_Dataset.current_labels = data_subset_labels

    @staticmethod
    def get(key='class_ids'):
        if key == 'data':
            return MNIST_Dataset.data
        elif key == 'data_numpy':
            return MNIST_Dataset.data_numpy
        elif key == 'data_indices':
            return MNIST_Dataset.data_indices
        elif key == 'data_labels':
            return MNIST_Dataset.data_labels
        elif key == 'classes':
            return MNIST_Dataset.classes
        elif key == 'class_ids':
            return MNIST_Dataset.class_ids
        elif key == 'original_subset_indices':
            return MNIST_Dataset.original_subset_indices
        elif key == 'current_subset_indices':
            return MNIST_Dataset.current_subset_indices
        elif key == 'current_subset':
            return MNIST_Dataset.current_subset
        elif key == 'current_subset_numpy':
            return MNIST_Dataset.current_subset_numpy
        elif key == 'current_labels':
            return MNIST_Dataset.current_labels
        else:
            raise ValueError('Invalid key passed to Dataset class get function')

    @staticmethod
    def download_and_transform_dataset(download=True):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_data = MNIST(root='./data', train=True, download=download, transform=transform)
        return mnist_data


class Fashion_MNIST_Dataset:
    data = None
    data_numpy = None
    data_indices = None
    data_labels = None
    classes = None
    class_ids = None
    original_subset_indices = None
    current_subset_indices = None
    current_subset = None
    current_subset_numpy = None
    current_labels = None

    @staticmethod
    def load_data(downloaded_data):
        Fashion_MNIST_Dataset.data = downloaded_data
        data_images = torch.stack([img[0].squeeze() for img in downloaded_data])
        data_images_flat = data_images.view(data_images.size(0), -1).numpy()
        Fashion_MNIST_Dataset.data_numpy = data_images_flat
        Fashion_MNIST_Dataset.data_indices = np.array(list(range(len(downloaded_data))))
        Fashion_MNIST_Dataset.classes = list(downloaded_data.classes)
        data_labels = [Fashion_MNIST_Dataset.classes[downloaded_data[i][1]] for i in range(len(downloaded_data))]
        Fashion_MNIST_Dataset.data_labels = data_labels
        Fashion_MNIST_Dataset.class_ids = list(range(len(Fashion_MNIST_Dataset.classes)))

    @staticmethod
    def get_data_subset(downloaded_data, sample_size=5000):
        sample_indices = random.sample(range(len(downloaded_data)), sample_size)
        Fashion_MNIST_Dataset.original_subset_indices = sample_indices
        Fashion_MNIST_Dataset.current_subset_indices = np.array(list(range(sample_size)))
        subset = Subset(downloaded_data, sample_indices)
        Fashion_MNIST_Dataset.current_subset = subset
        x = torch.stack([img[0].squeeze() for img in subset])
        x_flat = x.view(x.size(0), -1).numpy()
        labels = [Fashion_MNIST_Dataset.classes[downloaded_data[i][1]] for i in sample_indices]
        return x_flat, labels

    @staticmethod
    def load_current_subset(data_subset_numpy, data_subset_labels):
        Fashion_MNIST_Dataset.current_subset_numpy = data_subset_numpy
        Fashion_MNIST_Dataset.current_labels = data_subset_labels

    @staticmethod
    def get(key='class_ids'):
        if key == 'data':
            return Fashion_MNIST_Dataset.data
        elif key == 'data_numpy':
            return Fashion_MNIST_Dataset.data_numpy
        elif key == 'data_indices':
            return Fashion_MNIST_Dataset.data_indices
        elif key == 'data_labels':
            return Fashion_MNIST_Dataset.data_labels
        elif key == 'classes':
            return Fashion_MNIST_Dataset.classes
        elif key == 'class_ids':
            return Fashion_MNIST_Dataset.class_ids
        elif key == 'original_subset_indices':
            return Fashion_MNIST_Dataset.original_subset_indices
        elif key == 'current_subset_indices':
            return Fashion_MNIST_Dataset.current_subset_indices
        elif key == 'current_subset':
            return Fashion_MNIST_Dataset.current_subset
        elif key == 'current_subset_numpy':
            return Fashion_MNIST_Dataset.current_subset_numpy
        elif key == 'current_labels':
            return Fashion_MNIST_Dataset.current_labels
        else:
            raise ValueError('Invalid key passed to Dataset class get function')

    @staticmethod
    def download_and_transform_dataset(download=True):
        transform = transforms.Compose([transforms.ToTensor()])
        fashion_mnist_data = FashionMNIST(root='./data', train=True, download=download, transform=transform)
        return fashion_mnist_data


class CIFAR_100_Dataset:
    data = None
    data_numpy = None
    data_indices = None
    data_labels = None
    classes = None
    class_ids = None
    original_subset_indices = None
    current_subset_indices = None
    current_subset = None
    current_subset_numpy = None
    current_labels = None

    @staticmethod
    def load_data(downloaded_data):
        CIFAR_100_Dataset.data = downloaded_data
        data_images = torch.stack([img[0].squeeze() for img in downloaded_data])
        data_images_flat = data_images.view(data_images.size(0), -1).numpy()
        CIFAR_100_Dataset.data_numpy = data_images_flat
        CIFAR_100_Dataset.data_indices = np.array(list(range(len(downloaded_data))))
        CIFAR_100_Dataset.classes = list(downloaded_data.classes)
        data_labels = [CIFAR_100_Dataset.classes[downloaded_data[i][1]] for i in range(len(downloaded_data))]
        CIFAR_100_Dataset.data_labels = data_labels
        CIFAR_100_Dataset.class_ids = list(range(len(CIFAR_100_Dataset.classes)))

    @staticmethod
    def get_data_subset(downloaded_data, sample_size=5000):
        sample_indices = random.sample(range(len(downloaded_data)), sample_size)
        CIFAR_100_Dataset.original_subset_indices = sample_indices
        CIFAR_100_Dataset.current_subset_indices = np.array(list(range(sample_size)))
        subset = Subset(downloaded_data, sample_indices)
        CIFAR_100_Dataset.current_subset = subset
        x = torch.stack([img[0].squeeze() for img in subset])
        x_flat = x.view(x.size(0), -1).numpy()
        labels = [CIFAR_100_Dataset.classes[downloaded_data[i][1]] for i in sample_indices]
        return x_flat, labels

    @staticmethod
    def load_current_subset(data_subset_numpy, data_subset_labels):
        CIFAR_100_Dataset.current_subset_numpy = data_subset_numpy
        CIFAR_100_Dataset.current_labels = data_subset_labels

    @staticmethod
    def get(key='class_ids'):
        if key == 'data':
            return CIFAR_100_Dataset.data
        elif key == 'data_numpy':
            return CIFAR_100_Dataset.data_numpy
        elif key == 'data_indices':
            return CIFAR_100_Dataset.data_indices
        elif key == 'data_labels':
            return CIFAR_100_Dataset.data_labels
        elif key == 'classes':
            return CIFAR_100_Dataset.classes
        elif key == 'class_ids':
            return CIFAR_100_Dataset.class_ids
        elif key == 'original_subset_indices':
            return CIFAR_100_Dataset.original_subset_indices
        elif key == 'current_subset_indices':
            return CIFAR_100_Dataset.current_subset_indices
        elif key == 'current_subset':
            return CIFAR_100_Dataset.current_subset
        elif key == 'current_subset_numpy':
            return CIFAR_100_Dataset.current_subset_numpy
        elif key == 'current_labels':
            return CIFAR_100_Dataset.current_labels
        else:
            raise ValueError('Invalid key passed to Dataset class get function')

    @staticmethod
    def download_and_transform_dataset(download=True):
        transform = transforms.Compose([transforms.ToTensor()])
        cifar_data = CIFAR100(root='./data', train=True, download=download, transform=transform)
        return cifar_data
