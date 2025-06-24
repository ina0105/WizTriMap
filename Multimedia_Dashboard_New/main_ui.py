import argparse
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from trimap.trimap import TRIMAP
from dash import html, dcc
import torch
import warnings

from app import app
from widgets import header_tabs, dataset_dropdown, data_stores_and_triggers
import callbacks.clientside_callbacks
import callbacks.header_tabs
import callbacks.dataset_dropdown
import callbacks.dr_scatterplot

from helpers.models import ConvDecoderGray, ConvDecoder, get_inversion_from_model
from helpers import datasets
from helpers.cache import Cache, TriMapInversionCache, CNNLayerInversionCache
from helpers.config import dataset_list, method_list, dim_keys, cnn_layers, dataset_class_mapping

warnings.filterwarnings("ignore", category=FutureWarning)

pca_max_image_threshold_for_scale = 200
t_sne_max_image_threshold_for_scale = 200
umap_max_image_threshold_for_scale = 200
trimap_max_image_threshold_for_scale = 200

projection_method_fit_transform_dict = {
    "PCA": lambda x, y: PCA(n_components=y).fit_transform(x),
    "t_SNE": lambda x, y: TSNE(n_components=y, init='random', random_state=42).fit_transform(x),
    "UMAP": lambda x, y: umap.UMAP(n_components=y).fit_transform(x, njobs=-1),
    "TriMap": lambda x, y: TRIMAP(n_dims=y).fit_transform(x)
}


def preload_all(device, sample_size):
    for dataset in dataset_list:
        dataset_class = getattr(datasets, dataset_class_mapping[dataset])
        dataset_folder_name = 'cifar-100-python' if dataset == 'CIFAR-100' else dataset
        if not os.path.isdir(f'data/{dataset_folder_name}'):
            os.makedirs(f'data/{dataset_folder_name}')
            downloaded_data = dataset_class.download_and_transform_dataset()
        else:
            downloaded_data = dataset_class.download_and_transform_dataset(download=False)

        dataset_class.load_data(downloaded_data)
        x_flat, labels = dataset_class.get_data_subset(downloaded_data, sample_size=sample_size)
        dataset_class.load_current_subset(x_flat, labels)

        if not os.path.isdir(f'pre_loaded_embeddings/{dataset_folder_name}'):
            os.makedirs(f'pre_loaded_embeddings/{dataset_folder_name}')
            load_prefetched = False
        else:
            load_prefetched = True

        for method in method_list:
            if not load_prefetched:
                embedding_results = []
                inversion_results = []
                for model_input_dim in dim_keys[method]:
                    x_embedding = projection_method_fit_transform_dict[method](x_flat, model_input_dim)
                    with open(f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_embeddings', 'wb') as file:
                        pickle.dump(x_embedding, file)
                    embedding_results.append(x_embedding)
                    try:
                        if dataset == 'CIFAR-100':
                            inversion_model = ConvDecoder(input_dim=model_input_dim)
                            formatted_dataset_name = dataset.replace('-', '_')
                        else:
                            inversion_model = ConvDecoderGray(input_dim=model_input_dim)
                            formatted_dataset_name = dataset
                    except Exception as exception:
                        print('Train the Inversion model or copy and paste the multi_dimension_inversion_models folder '
                              'from <URL> in the working directory / main folder')
                        raise exception
                    inversion_model.load_state_dict(torch.load(
                        f'multi_dimension_inversion_models/{formatted_dataset_name}_{method}_{model_input_dim}D.pth'))
                    inversion_model.to(device)
                    x_inversion = get_inversion_from_model(inversion_model, x_embedding, sample_size, device)
                    with open(f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_inversion', 'wb') as file:
                        pickle.dump(x_inversion, file)
                    inversion_results.append(x_inversion)
            else:
                embedding_results = []
                inversion_results = []
                for model_input_dim in dim_keys[method]:
                    with open(f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_embeddings', 'rb') as file:
                        x_embedding = pickle.load(file)

                    with open(f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_inversion', 'rb') as file:
                        x_inversion = pickle.load(file)

                    embedding_results.append(x_embedding)
                    inversion_results.append(x_inversion)

            Cache.load_embedding_cache(dataset, method, embedding_results[0])
            Cache.load_last_embeddings(dataset, method, Cache.get_embedding_cache(dataset, method))
            model, x_recon = inversion_results[0]
            Cache.load_model_cache(dataset, method, (model, x_recon))

            if method == 'TriMap':
                for input_dim_index, model_input_dim in enumerate(dim_keys[method]):
                    TriMapInversionCache.load_embedding_cache(dataset, str(model_input_dim),
                                                              embedding_results[input_dim_index])
                    TriMapInversionCache.load_model_cache(dataset, str(model_input_dim),
                                                          inversion_results[input_dim_index])

                    try:
                        if dataset == 'CIFAR-100':
                            cnn_layer_inversion_model = ConvDecoder(input_dim=2)
                            formatted_dataset_name = dataset.replace('-', '_')
                        else:
                            cnn_layer_inversion_model = ConvDecoderGray(input_dim=2)
                            formatted_dataset_name = dataset
                    except Exception as exception:
                        print('Train the Inversion models for CNN Layers or copy and paste the '
                              'cnn_layer_embeddings_trimap folder from <URL> in the working directory / main folder')
                        raise exception

                    for layer_number in cnn_layers:
                        layer_embeddings = torch.load(f'cnn_layer_embeddings_trimap/{formatted_dataset_name}_{method}_layer{layer_number}_embeddings.pt')
                        CNNLayerInversionCache.load_embedding_cache(dataset, str(layer_number), layer_embeddings)

                        if not load_prefetched:
                            cnn_layer_inversion_model.load_state_dict(torch.load(
                                f'cnn_layer_inversion_models/{formatted_dataset_name}_{method}_layer{layer_number}.pth',
                                weights_only=False))
                            cnn_layer_inversion_model.to(device)
                            x_inversion = get_inversion_from_model(cnn_layer_inversion_model, layer_embeddings,
                                                                   layer_embeddings.shape[0], device)
                            with open(
                                    f'pre_loaded_embeddings/{dataset_folder_name}/{method}_cnn_{layer_number}_inversion',
                                    'wb') as file:
                                pickle.dump(x_inversion, file)
                        else:
                            with open(
                                    f'pre_loaded_embeddings/{dataset_folder_name}/{method}_cnn_{layer_number}_inversion',
                                    'rb') as file:
                                x_inversion = pickle.load(file)
                        model, x_recon = x_inversion
                        CNNLayerInversionCache.load_model_cache(dataset, str(layer_number), (model, x_recon))


def run_ui():
    header_tab_widget = header_tabs.create_header_tabs()
    dataset_dropdown_widget = dataset_dropdown.create_dataset_dropdown()
    data_stores_and_triggers_area = data_stores_and_triggers.create_data_stores_and_triggers()

    app.layout = html.Div([
        html.Div([
            header_tab_widget,
            dataset_dropdown_widget],
            id='header-bar'
        ),
        dcc.Loading(id='dashboard-loading', children=dcc.Store(id='selected-tab'), type='circle'),
        html.Div(id='main-content', className='main-content-area'),
        data_stores_and_triggers_area,
        dcc.Interval(id='init-trigger', interval=100, max_intervals=1, disabled=False),
        dcc.Store(id='init-load', data=False)
    ])

    app.run(debug=True, use_reloader=True)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isdir('data'):
        os.makedirs('data')

    if not os.path.isdir('pre_loaded_embeddings'):
        os.makedirs('pre_loaded_embeddings')

    preload_all(device, args.sample_size)

    run_ui()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', default=5000, type=int, help='Sample Size to consider')

    parser_args = parser.parse_args()
    main(parser_args)
