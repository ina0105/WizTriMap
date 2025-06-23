# WizTriMap

In this work, we aim to enhance TriMap by making it more expressive and scalable for practical applications. Our target users include AI practitioners, developers, and machine learning researchers who seek to visually analyze latent spaces, explore data structures, and assess how classifiers separate different classes. To address these needs, we present a user-centered extension of TriMap that introduces new functionalities.

https://github.com/user-attachments/assets/30385386-96bc-4f97-9dcc-ddb04c1f1866

## Project Structure

```
WizTriMap
├──
├── utils
│   ├── utils.py
│   └── models.py
├── cnn
│   ├── extract_embeddings_cnn.py
│   └── train_inversion_models_cnn.py
├── initial data
│   ├── train_inversion_models.py
│   ├── save_projections.py
│   └── run_intference.py
├── additional experiments
│   ├── hyperbolic.py
│   └── hyperbolic-corr-fashionmnist.py
├── WizTrimap_env.yml
└── README.md
```
## Datasets
1. MNIST
    The MNIST database of handwritten digits is one of the most popular image recognition datasets. It contains 60k examples for training and 10k examples for testing. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
2. FashionMNIST
    Fashion-MNIST is a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
3. CIFAR-100
    The CIFAR 100 dataset is commonly used for image classification and recognition. The CIFAR-100 dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class. There are 50000 training images and 10000 test images. 

## Installation

To install the environment with the required dependencies, run:
```
conda env create -f WizTrimap_env.yml
conda activate WizTrimap
```

## Full Pipeline of the Demo

To access the results as shown in the Demo presentation of this system, one must obtain the necessary data in advance; otherwise, the execution time would pose a constraint. The pipeline to the Demo is as follows:
#### 1. Initial Data Processing
   1.1 Load and normalize the train data for MNIST, FashionMNIST, and CIFAR-100 using torchvision \
   1.2 For each sample, compute latent embeddings on-the-fly (TriMap: 2D–7D, others: only 2D) \
   1.3 Train inversion models to reconstruct images from embeddings and save the best model per dataset-method-dim in `saved_models/` \
   1.4 Compute and save reconstruction errors as `.npy` files and 20 reconstructed image samples in `recon_output/<dataset>/<method>_<dim>D/`

#### 2. CNN visualisation 
   2.1 Extracting embeddings from three layers of a simple CNN for every Dataset \
   2.2 Training inversion models to invert from those embeddings to the original image
#### 3. Running the Demo
   3.1


## Additional Dataset
The code can easily be updated to accommodate other datasets, keeping in mind that adding a dataset cannot be visualized in real-time as the computation of embeddings and the training of the models will take a prolonged period of time.

## Usage
### CNN
```bash
python cnn/extract_embeddings_cnn.py
python cnn/train_inversion_models_cnn.py
```
### Initial Data
### Running the Demo
