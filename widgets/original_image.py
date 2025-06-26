from dash import html, dcc
from PIL import Image
import numpy as np

from helpers import datasets
from utils.utils import pil_to_base64, get_blank_image_base64
from helpers.config import dataset_class_mapping, euclidean_image_resize_value, \
    multi_recon_original_image_resize_value, model_prog_image_resize_value


def display_euclidean_original_image(click_data, dataset, scatterplot_fig):
    dataset_class = getattr(datasets, dataset_class_mapping[dataset])
    point = click_data['points'][0]
    x_val = point['x']
    y_val = point['y']

    scatterplot_data = scatterplot_fig['data'][0]
    scatter_image_ids = [int(row[0]) for row in scatterplot_data['customdata']]
    scatter_x = scatterplot_data['x']
    scatter_y = scatterplot_data['y']

    original_image_id = -1
    for x, y, image_id in zip(scatter_x, scatter_y, scatter_image_ids):
        if x_val == x and y_val == y:
            original_image_id = image_id
            break

    if original_image_id == -1:
        print('x, y not matched')

    original_image_from_id = dataset_class.get(key='current_subset')[original_image_id][0]
    original_image_array = original_image_from_id.numpy()
    if dataset == 'CIFAR-100':
        original_image_array = np.transpose(original_image_array, (1, 2, 0))
        original_image_array = np.clip(original_image_array * 255, 0, 255).astype(np.uint8)
        original_image = pil_to_base64(Image.fromarray(original_image_array).resize(euclidean_image_resize_value,
                                                                                    resample=Image.NEAREST))
    else:
        original_image_array = original_image_array.squeeze(0)
        original_image_array = np.clip(original_image_array * 255, 0, 255).astype(np.uint8)
        original_image = pil_to_base64(Image.fromarray(original_image_array, mode='L').resize(
            euclidean_image_resize_value, resample=Image.NEAREST))
    return original_image


def display_multi_recon_original_image(click_data, dataset, scatterplot_fig):
    dataset_class = getattr(datasets, dataset_class_mapping[dataset])
    point = click_data['points'][0]
    x_val = point['x']
    y_val = point['y']

    scatterplot_data = scatterplot_fig['data'][0]
    scatter_image_ids = [int(row[0]) for row in scatterplot_data['customdata']]
    scatter_x = scatterplot_data['x']
    scatter_y = scatterplot_data['y']

    original_image_id = -1
    for x, y, image_id in zip(scatter_x, scatter_y, scatter_image_ids):
        if x_val == x and y_val == y:
            original_image_id = image_id
            break

    if original_image_id == -1:
        print('x, y not matched')

    original_image_from_id = dataset_class.get(key='current_subset')[original_image_id][0]
    original_image_array = original_image_from_id.numpy()
    if dataset == 'CIFAR-100':
        original_image_array = np.transpose(original_image_array, (1, 2, 0))
        original_image_array = np.clip(original_image_array * 255, 0, 255).astype(np.uint8)
        original_image = pil_to_base64(Image.fromarray(original_image_array).resize(
            multi_recon_original_image_resize_value, resample=Image.NEAREST))
    else:
        original_image_array = original_image_array.squeeze(0)
        original_image_array = np.clip(original_image_array * 255, 0, 255).astype(np.uint8)
        original_image = pil_to_base64(Image.fromarray(original_image_array, mode='L').resize(
            multi_recon_original_image_resize_value, resample=Image.NEAREST))
    return original_image


def display_cnn_layer_emb_original_image(click_data, dataset, scatterplot_fig):
    dataset_class = getattr(datasets, dataset_class_mapping[dataset])
    point = click_data['points'][0]
    x_val = point['x']
    y_val = point['y']

    scatterplot_data = scatterplot_fig['data'][0]
    scatter_image_ids = [int(row[0]) for row in scatterplot_data['customdata']]
    scatter_x = scatterplot_data['x']
    scatter_y = scatterplot_data['y']

    original_image_id = -1
    for x, y, image_id in zip(scatter_x, scatter_y, scatter_image_ids):
        if x_val == x and y_val == y:
            original_image_id = image_id
            break

    if original_image_id == -1:
        print('x, y not matched')

    original_image_from_id = dataset_class.get(key='current_subset')[original_image_id][0]
    original_image_array = original_image_from_id.numpy()
    if dataset == 'CIFAR-100':
        original_image_array = np.transpose(original_image_array, (1, 2, 0))
        original_image_array = np.clip(original_image_array * 255, 0, 255).astype(np.uint8)
        original_image = pil_to_base64(Image.fromarray(original_image_array).resize(model_prog_image_resize_value,
                                                                                    resample=Image.NEAREST))
    else:
        original_image_array = original_image_array.squeeze(0)
        original_image_array = np.clip(original_image_array * 255, 0, 255).astype(np.uint8)
        original_image = pil_to_base64(Image.fromarray(original_image_array, mode='L').resize(
            model_prog_image_resize_value, resample=Image.NEAREST))
    return original_image


def create_euclidean_original_image_widget():
    return html.Div([dcc.Loading(id='loading-original-image-euclidean', children=[
        html.Img(id='original-image-euclidean', src=get_blank_image_base64())], type='circle')],
                    id='original-image-div-euclidean')


def create_multi_recon_original_image_widget():
    return html.Div([dcc.Loading(id='loading-original-image-multi-recon', children=[
        html.Img(id='original-image-multi-recon', src=get_blank_image_base64())], type='circle')],
                    id='original-image-div-multi-recon')


def create_model_prog_original_image_widget():
    return html.Div([dcc.Loading(id='loading-original-image-model-prog', children=[
        html.Img(id='original-image-model-prog', src=get_blank_image_base64())], type='circle')],
                    id='original-image-div-model-prog')
