from dash import html, dcc
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np
import torch

from helpers.cache import Cache, TriMapInversionCache, CNNLayerInversionCache
from helpers.utils import pil_to_base64, get_blank_image_base64
from helpers.config import euclidean_image_resize_value, multi_recon_recon_image_resize_value, \
    model_prog_image_resize_value, dim_keys


def display_euclidean_recon_image(click_data, dataset, method):
    if click_data.get('points'):
        point = click_data['points'][0]
        x_val = point['x']
        y_val = point['y']
    else:
        x_val = click_data['x']
        y_val = click_data['y']

    model, _ = Cache.get_model_cache(dataset, method)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([[x_val, y_val]], dtype=torch.float32).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).cpu().numpy()

    if dataset == 'CIFAR-100':
        out = out.squeeze(0)
        out = np.transpose(out, (1, 2, 0))
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        recon_image = pil_to_base64(Image.fromarray(out).resize(euclidean_image_resize_value, resample=Image.NEAREST))
    else:
        out = out.reshape(28, 28)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        recon_image = pil_to_base64(Image.fromarray(out, mode='L').resize(euclidean_image_resize_value,
                                                                          resample=Image.NEAREST))
    return recon_image


def display_multi_recon_recon_image(click_data, dataset, method):
    point = click_data['points'][0]
    x_val = point['x']
    y_val = point['y']

    anchor_embeddings = TriMapInversionCache.get_embedding_cache(dataset, '2')
    datapoint_actual_index = -1
    for datapoint_index in range(anchor_embeddings.shape[0]):
        if x_val == anchor_embeddings[datapoint_index][0] and y_val == anchor_embeddings[datapoint_index][1]:
            datapoint_actual_index = datapoint_index
            break

    if datapoint_actual_index == -1:
        print('No datapoint found')
        raise Exception('No datapoint found')

    recon_image_list = []
    for method_input_dim in dim_keys[method]:
        embeddings = TriMapInversionCache.get_embedding_cache(dataset, str(method_input_dim))
        model, _ = TriMapInversionCache.get_model_cache(dataset, str(method_input_dim))
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(embeddings[datapoint_actual_index].reshape(1, -1), dtype=torch.float32).to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).cpu().numpy()

        if dataset == 'CIFAR-100':
            out = out.squeeze(0)
            out = np.transpose(out, (1, 2, 0))
            out = np.clip(out * 255, 0, 255).astype(np.uint8)
            recon_image = pil_to_base64(Image.fromarray(out).resize(multi_recon_recon_image_resize_value,
                                                                    resample=Image.NEAREST))
        else:
            out = out.reshape(28, 28)
            out = np.clip(out * 255, 0, 255).astype(np.uint8)
            recon_image = pil_to_base64(Image.fromarray(out, mode='L').resize(multi_recon_recon_image_resize_value,
                                                                              resample=Image.NEAREST))
        recon_image_list.append(recon_image)
    return recon_image_list


def display_cnn_layer_emb_recon_image(click_data, dataset, layer_number):
    if click_data.get('points'):
        point = click_data['points'][0]
        x_val = point['x']
        y_val = point['y']
    else:
        x_val = click_data['x']
        y_val = click_data['y']

    model, _ = CNNLayerInversionCache.get_model_cache(dataset, str(layer_number))
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([[x_val, y_val]], dtype=torch.float32).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).cpu().numpy()

    if dataset == 'CIFAR-100':
        out = out.squeeze(0)
        out = np.transpose(out, (1, 2, 0))
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        recon_image = pil_to_base64(Image.fromarray(out).resize(model_prog_image_resize_value, resample=Image.NEAREST))
    else:
        out = out.reshape(28, 28)
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        recon_image = pil_to_base64(Image.fromarray(out, mode='L').resize(model_prog_image_resize_value,
                                                                          resample=Image.NEAREST))
    return recon_image


def create_euclidean_recon_image_widget():
    return html.Div([dcc.Loading(id='loading-recon-image-euclidean', children=[html.Img(id='recon-image-euclidean',
                                                                                        src=get_blank_image_base64())],
                                 type='circle')], id='recon-image-div-euclidean')


def create_multi_recon_image_grid():
    return dbc.Row([
            dbc.Col([
                html.Div("TriMap-2D", className='panel-label'),
                html.Div([
                    dcc.Loading(id='loading-multi-recon-image-2d', children=[html.Img(id='multi-recon-image-2d',
                                                                                      src=get_blank_image_base64())],
                                type='circle')
                ], id='multi-recon-image-div-2d')
            ], className='multi-recon-image-grid-column border-widget', style={'width': '14%'}),
            dbc.Col([
                html.Div("TriMap-3D", className='panel-label'),
                html.Div([
                    dcc.Loading(id='loading-multi-recon-image-3d', children=[html.Img(id='multi-recon-image-3d',
                                                                                      src=get_blank_image_base64())],
                                type='circle')
                ], id='multi-recon-image-div-3d')
            ], className='multi-recon-image-grid-column border-widget', style={'width': '14%'}),
            dbc.Col([
                html.Div("TriMap-4D", className='panel-label'),
                html.Div([
                    dcc.Loading(id='loading-multi-recon-image-4d', children=[html.Img(id='multi-recon-image-4d',
                                                                                      src=get_blank_image_base64())],
                                type='circle')
                ], id='multi-recon-image-div-4d')
            ], className='multi-recon-image-grid-column border-widget', style={'width': '14%'}),
            dbc.Col([
                html.Div("TriMap-5D", className='panel-label'),
                html.Div([
                    dcc.Loading(id='loading-multi-recon-image-5d', children=[html.Img(id='multi-recon-image-5d',
                                                                                      src=get_blank_image_base64())],
                                type='circle')
                ], id='multi-recon-image-div-5d')
            ], className='multi-recon-image-grid-column border-widget', style={'width': '14%'}),
            dbc.Col([
                html.Div("TriMap-6D", className='panel-label'),
                html.Div([
                    dcc.Loading(id='loading-multi-recon-image-6d', children=[html.Img(id='multi-recon-image-6d',
                                                                                      src=get_blank_image_base64())],
                                type='circle')
                ], id='multi-recon-image-div-6d')
            ], className='multi-recon-image-grid-column border-widget', style={'width': '14%'}),
            dbc.Col([
                html.Div("TriMap-7D", className='panel-label'),
                html.Div([
                    dcc.Loading(id='loading-multi-recon-image-7d', children=[html.Img(id='multi-recon-image-7d',
                                                                                      src=get_blank_image_base64())],
                                type='circle')
                ], id='multi-recon-image-div-7d')
            ], className='multi-recon-image-grid-last-column border-widget', style={'width': '14%'})
        ], className='multi-recon-grid-row', justify='between')


def create_model_prog_recon_image_widget():
    return html.Div([dcc.Loading(id='loading-recon-image-model-prog', children=[html.Img(id='recon-image-model-prog',
                                                                                         src=get_blank_image_base64())],
                                 type='circle')], id='recon-image-div-model-prog')
