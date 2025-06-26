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
from widgets import header_tabs, logo_widget, dataset_dropdown, data_stores_and_triggers
import callbacks.clientside_callbacks
import callbacks.header_tabs
import callbacks.dataset_dropdown
import callbacks.dr_scatterplot

from models.models import ConvDecoderGray, ConvDecoder, get_inversion_from_model
from helpers import datasets
from helpers.cache import Cache, TriMapInversionCache, CNNLayerInversionCache
from helpers.config import dataset_list, method_list, dim_keys, cnn_layers, dataset_class_mapping, cache_file_paths

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
        print(f'Loading Dataset: {dataset}')
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
        print(f'Dataset {dataset} loaded successfully')

    cache_exists = True
    for cache_file_path in cache_file_paths:
        if not os.path.exists(cache_file_path):
            cache_exists = False
            break

    if not cache_exists:
        print('Saved Cache not found. Calculating required data.')
        for dataset in dataset_list:
            print(f'Starting Calculation for Dataset: {dataset}')
            dataset_class = getattr(datasets, dataset_class_mapping[dataset])
            dataset_folder_name = 'cifar-100-python' if dataset == 'CIFAR-100' else dataset
            if not os.path.isdir(f'pre_loaded_embeddings/{dataset_folder_name}'):
                os.makedirs(f'pre_loaded_embeddings/{dataset_folder_name}')

            x_flat = dataset_class.get(key='current_subset_numpy')
            for method in method_list:
                print(f'Resolving Method: {method}')
                embedding_results = []
                inversion_results = []
                for model_input_dim in dim_keys[method]:
                    if os.path.exists(f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_embeddings.npy'):
                        x_embedding = np.load(f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_embeddings.npy')
                    else:
                        x_embedding = projection_method_fit_transform_dict[method](x_flat, model_input_dim)
                        np.save(f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_embeddings.npy',
                                x_embedding)
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
                    if os.path.exists(
                            f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_recon_result.npy'):
                        x_recon = np.load(f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_recon_result.npy')
                    else:
                        x_recon = get_inversion_from_model(inversion_model, x_embedding, sample_size, device)
                        np.save(
                            f'pre_loaded_embeddings/{dataset_folder_name}/{method}_{model_input_dim}_recon_result.npy',
                            x_recon)
                    inversion_results.append((inversion_model, x_recon))

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
                            layer_embeddings = torch.load(
                                f'cnn_layer_embeddings_trimap/{formatted_dataset_name}_{method}_layer{layer_number}_embeddings.pt',
                                weights_only=False)
                            layer_embeddings = np.array(
                                [layer_embeddings[i, :] for i in dataset_class.get(key='original_subset_indices')])
                            CNNLayerInversionCache.load_embedding_cache(dataset, str(layer_number), layer_embeddings)

                            cnn_layer_inversion_model.load_state_dict(torch.load(
                                f'cnn_layer_inversion_models/{formatted_dataset_name}_{method}_layer{layer_number}.pth',
                                weights_only=False))
                            cnn_layer_inversion_model.to(device)

                            if os.path.exists(
                                    f'pre_loaded_embeddings/{dataset_folder_name}/{method}_cnn_{layer_number}_recon_result.npy'):
                                x_recon = np.load(
                                    f'pre_loaded_embeddings/{dataset_folder_name}/{method}_cnn_{layer_number}_recon_result.npy')
                            else:
                                x_recon = get_inversion_from_model(cnn_layer_inversion_model, layer_embeddings,
                                                                   sample_size, device)
                                np.save(
                                    f'pre_loaded_embeddings/{dataset_folder_name}/{method}_cnn_{layer_number}_recon_result.npy',
                                    x_recon)
                            CNNLayerInversionCache.load_model_cache(dataset, str(layer_number),
                                                                    (cnn_layer_inversion_model, x_recon))
                print(f'Method {method} done')
            print(f'Calculation for Dataset {dataset} done')

        print('Saving Cache classes')

        with open('cache_folder/cache_class_embedding_cache.pickle', 'wb') as file:
            pickle.dump(Cache.embedding_cache, file)

        with open('cache_folder/cache_class_model_cache.pickle', 'wb') as file:
            pickle.dump(Cache.model_cache, file)

        with open('cache_folder/trimap_inversion_cache_class_embedding_cache.pickle', 'wb') as file:
            pickle.dump(TriMapInversionCache.embedding_cache, file)

        with open('cache_folder/trimap_inversion_cache_class_model_cache.pickle', 'wb') as file:
            pickle.dump(TriMapInversionCache.model_cache, file)

        with open('cache_folder/cnn_inversion_cache_class_embedding_cache.pickle', 'wb') as file:
            pickle.dump(CNNLayerInversionCache.embedding_cache, file)

        with open('cache_folder/cnn_inversion_cache_class_model_cache.pickle', 'wb') as file:
            pickle.dump(CNNLayerInversionCache.model_cache, file)

        print('Cache classes saved successfully')
    else:
        print('Found saved Cache')
        print('Loading Cache classes')
        with open('cache_folder/cache_class_embedding_cache.pickle', 'rb') as file:
            Cache.embedding_cache = pickle.load(file)

        with open('cache_folder/cache_class_model_cache.pickle', 'rb') as file:
            Cache.model_cache = pickle.load(file)

        with open('cache_folder/trimap_inversion_cache_class_embedding_cache.pickle', 'rb') as file:
            TriMapInversionCache.embedding_cache = pickle.load(file)

        with open('cache_folder/trimap_inversion_cache_class_model_cache.pickle', 'rb') as file:
            TriMapInversionCache.model_cache = pickle.load(file)

        with open('cache_folder/cnn_inversion_cache_class_embedding_cache.pickle', 'rb') as file:
            CNNLayerInversionCache.embedding_cache = pickle.load(file)

        with open('cache_folder/cnn_inversion_cache_class_model_cache.pickle', 'rb') as file:
            CNNLayerInversionCache.model_cache = pickle.load(file)

        print('Cache classes loaded successfully')


def run_ui():
    header_tab_widget = header_tabs.create_header_tabs()
    logo_image_widget = logo_widget.create_logo_image_widget()
    dataset_dropdown_widget = dataset_dropdown.create_dataset_dropdown()
    data_stores_and_triggers_area = data_stores_and_triggers.create_data_stores_and_triggers()

    app.layout = html.Div([
        html.Div([
            header_tab_widget,
            logo_image_widget,
            dataset_dropdown_widget],
            id='header-bar'
        ),
        dcc.Loading(id='dashboard-loading', children=dcc.Store(id='selected-tab'), type='circle'),
        html.Div(id='main-content', className='main-content-area'),
        data_stores_and_triggers_area,
        dcc.Interval(id='init-trigger', interval=100, max_intervals=1, disabled=False),
        dcc.Store(id='init-load', data=False)
    ])

    app.run(debug=True, use_reloader=False)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isdir('data'):
        os.makedirs('data')

    if not os.path.isdir('pre_loaded_embeddings'):
        os.makedirs('pre_loaded_embeddings')

    if not os.path.isdir('cache_folder'):
        os.makedirs('cache_folder')

    preload_all(device, args.sample_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', default=10000, type=int, help='Sample Size to consider')

    parser_args = parser.parse_args()
    main(parser_args)

    run_ui()
