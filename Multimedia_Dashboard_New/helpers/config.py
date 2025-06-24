default_projection = 'TriMap'

dataset_list = ['MNIST', 'FashionMNIST', 'CIFAR-100']
method_list = ['PCA', 't_SNE', 'UMAP', 'TriMap']
dim_keys = {'TriMap': [2, 3, 4, 5, 6, 7], 'UMAP': [2], 't_SNE': [2], 'PCA': [2]}
cnn_layers = [0, 1, 2]

dataset_class_mapping = {
    'MNIST': 'MNIST_Dataset',
    'FashionMNIST': 'Fashion_MNIST_Dataset',
    'CIFAR-100': 'CIFAR_100_Dataset'
}

euclidean_scatterplot_width = 600
euclidean_scatterplot_height = 263

multi_recon_scatterplot_width = 1075
multi_recon_scatterplot_height = 377.5

model_prog_scatterplot_width = 1075
model_prog_scatterplot_height = 495

logo_image_resize_value = (550, 250)
euclidean_image_resize_value = (150, 150)
multi_recon_original_image_resize_value = (250, 250)
multi_recon_recon_image_resize_value = (150, 150)
model_prog_image_resize_value = (200, 200)

max_images_on_scatterplot = 100
