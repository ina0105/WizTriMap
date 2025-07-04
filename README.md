# WizTriMap

In this work, we aim to enhance TriMap by making it more expressive and scalable for practical applications. Our target users include AI practitioners, developers, and machine learning researchers who seek to visually analyze latent spaces, explore data structures, and assess how classifiers separate different classes. To address these needs, we present a user-centered extension of TriMap that introduces new functionalities.

https://github.com/user-attachments/assets/c75b4f4e-912e-48a6-a635-1e4f115c5ffa
## Project Structure

```
WizTriMap
├── additional_experiments
│    ├── hyperbolic.ipynb, 
│    ├── hyperbolic-corr-fashionmnist.ipynb
│    ├── case_study_cifar100.ypnb
│    └── embeddings_EV.ypnb
├── assets
│    ├── clientside.js
│    └── main.css
├── callbacks
│    ├── clientside_callbacks.py
│    ├── dataset_dropdown.py
│    ├── dr_scatterplot.py
│    └── header_tabs.py
├── dashboards
│    ├── euclidean.py
│    ├── model_prog.py
│    └── multi_recon_trimap.py
├── helpers
│    ├── cache.py
│    ├── config.py
│    ├── datasets.py
│    └── fine_to_coarse.json
├── logo_image
│    ├── WizTriMap.png
├── models
│    ├── models.py
├── preprocessing
│    ├── extract_embeddings_cnn.py
│    ├── run_inference.py
│    ├── save_projections.py
│    ├── train_inversion_models.py
│    └── train_inversion_models_cnn.py
├── trimap
│    ├── trimap.py
├── utils
│   ├── utils.py
│   └── save_projections_utils.py
├── widgets
│    ├── data_stores_and_triggers.py
│    ├── dataset_dropdown.py
│    ├── dr_scatterplot.py
│    ├── header_tabs.py
│    ├── logo_widget.py
│    ├── original_image.py
│    ├── radio_button.py
│    └── recon_image.py
├── app.py
├── main_ui.py
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
   <br>
   1.2 For each sample, compute latent embeddings on-the-fly (TriMap: 2D–7D, others: only 2D) \
   <br>
   1.3 Train inversion models to reconstruct images from embeddings and save the best model per dataset-method-dim in `../multi_dimension_inversion_models/` \
   <br>
   1.4 Compute and save reconstruction errors as `.npy` files and 20 reconstructed image samples in `recon_output/<dataset>/<method>_<dim>D/`

#### 2. CNN visualisation 
   2.1 Extracting embeddings from three layers of a simple CNN for every Dataset \
   <br>
   2.2 Training inversion models to invert from those embeddings to the original image

#### 3. Running the Demo
   3.1 Either run the above two steps to train and save the models or copy the saved models 
   from the link <a>https://drive.google.com/drive/folders/1z4xYzHzD1oiJcL_TKMwGOQyzXkS0nQSP?usp=drive_link
   <br><br>
   After this, the main ui is ready to run. \
   <br>
   3.2 On the first run, the embeddings for each DR method will be computed and saved 
   inside a folder in the main working directory. This would ensure faster reload for 
   subsequent runs.


## Additional Dataset
The code can easily be updated to accommodate other datasets, keeping in mind that adding a dataset cannot be visualized in real-time as the computation of embeddings and the training of the models will take a prolonged period of time.

## Usage
To run the dashboard, either download the saved models and CNN embeddings from <a>https://drive.google.com/drive/folders/1z4xYzHzD1oiJcL_TKMwGOQyzXkS0nQSP?usp=drive_link and follow the steps under 'Running the Demo' or run the CNN and Initial Data steps, which will save the models and embeddings and then run the demo.

### CNN
```bash
python preprocessing/extract_embeddings_cnn.py
python preprocessing/train_inversion_models_cnn.py
```
### Initial Data
Run the 'run_inference.py' for inference, which will save 20 random reconstructed images and reconstructed errors of the test data as .npy file.
```bash
python preprocessing/train_inversion_models.py
python preprocessing/run_inference.py
```
### Running the Demo
Run the 'main_ui.py' for running the main dash app.
```bash
python main_ui.py
```
