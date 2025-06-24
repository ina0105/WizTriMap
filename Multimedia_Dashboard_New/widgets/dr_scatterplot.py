import numpy as np
from PIL import Image
from dash import dcc
import plotly.graph_objects as go

from helpers import datasets
from helpers.cache import Cache, CNNLayerInversionCache
from helpers.utils import pil_to_base64
from helpers.config import max_images_on_scatterplot, dataset_class_mapping


def add_images_to_scatterplot(dataset, scatterplot_fig, dashboard):
    dataset_class = getattr(datasets, dataset_class_mapping[dataset])
    scatterplot_fig['layout']['images'] = []
    scatterplot_data = scatterplot_fig['data'][0]
    scatter_image_ids = [int(row[0]) for row in scatterplot_data['customdata']]
    scatter_x = scatterplot_data['x']
    scatter_y = scatterplot_data['y']

    min_x, max_x = scatterplot_fig['layout']['xaxis']['range']
    min_y, max_y = scatterplot_fig['layout']['yaxis']['range']

    images_in_zoom = []
    for x, y, image_id in zip(scatter_x, scatter_y, scatter_image_ids):
        if min_x <= x <= max_x and min_y <= y <= max_y:
            images_in_zoom.append((x, y, image_id))
        if len(images_in_zoom) > max_images_on_scatterplot:
            return scatterplot_fig

    if images_in_zoom:
        for x, y, image_id in images_in_zoom:
            if dashboard == 'model_prog':
                image_from_id = dataset_class.get(key='data')[image_id][0]
            else:
                image_from_id = dataset_class.get(key='current_subset')[image_id][0]
            img_array = image_from_id.numpy()
            if dataset == 'CIFAR-100':
                img_array = np.transpose(img_array, (1, 2, 0)) * 255
                img_array = img_array.astype(np.uint8)
                source_image = pil_to_base64(Image.fromarray(img_array))
            else:
                img_array = img_array.squeeze(0) * 255
                img_array = img_array.astype(np.uint8)
                source_image = pil_to_base64(Image.fromarray(img_array, mode='L'))
            scatterplot_fig['layout']['images'].append(dict(
                x=x,
                y=y,
                source=source_image,
                xref='x',
                yref='y',
                sizex=0.2,
                sizey=0.2,
                xanchor='center',
                yanchor='middle',
                layer='above'
            ))
        return scatterplot_fig
    return scatterplot_fig


def create_scatterplot_figure_euclidean(dataset, projection, width, height):
    dataset_class = getattr(datasets, dataset_class_mapping[dataset])
    embedding = Cache.get_embedding_cache(dataset, projection)
    Cache.load_last_embeddings(dataset, projection, embedding)
    model, x_recon = Cache.get_model_cache(dataset, projection)
    recon_error = np.mean((dataset_class.get('current_subset_numpy') - x_recon) ** 2, axis=1)

    fig = go.Figure()
    indices = dataset_class.get(key='current_subset_indices')
    labels = dataset_class.get(key='current_labels')
    xy = np.stack([embedding[:, 0], embedding[:, 1]], axis=1)
    customdata = np.column_stack((indices, labels, xy))

    fig.add_trace(go.Scattergl(
        x=embedding[:, 0], y=embedding[:, 1], mode='markers',
        marker=dict(color=recon_error, colorscale='Inferno', size=2, colorbar=dict(title='Recon Error', thickness=10,
                                                                                   len=0.6, x=1.02)),
        name=projection,
        showlegend=False,
        customdata=customdata,
        hovertemplate='Latent X: %{customdata[2]:.2f}<br>Latent Y: %{customdata[3]:.2f}<br>Label: %{customdata[1]}'
    ))

    fig.update_layout(
        margin=dict(l=5, r=2, t=10, b=2),
        autosize=False,
        width=width,
        height=height
    )
    return fig


def create_scatterplot_figure_multi_recon(dataset, projection, width, height):
    dataset_class = getattr(datasets, dataset_class_mapping[dataset])
    embedding = Cache.get_embedding_cache(dataset, projection)
    Cache.load_last_embeddings(dataset, projection, embedding)
    model, x_recon = Cache.get_model_cache(dataset, projection)
    recon_error = np.mean((dataset_class.get('current_subset_numpy') - x_recon) ** 2, axis=1)

    fig = go.Figure()
    indices = dataset_class.get(key='current_subset_indices')
    labels = dataset_class.get(key='current_labels')
    xy = np.stack([embedding[:, 0], embedding[:, 1]], axis=1)
    customdata = np.column_stack((indices, labels, xy))

    fig.add_trace(go.Scattergl(
        x=embedding[:, 0], y=embedding[:, 1], mode='markers',
        marker=dict(color=recon_error, colorscale='Inferno', size=2, colorbar=dict(title='Recon Error', thickness=10,
                                                                                   len=0.6, x=1.02)),
        name=projection,
        showlegend=False,
        customdata=customdata,
        hovertemplate='Latent X: %{customdata[2]:.2f}<br>Latent Y: %{customdata[3]:.2f}<br>Label: %{customdata[1]}'
    ))

    fig.update_layout(
        margin=dict(l=5, r=2, t=10, b=2),
        autosize=False,
        width=width,
        height=height,
    )
    return fig


def create_scatterplot_figure_model_prog(dataset, layer_number, width, height):
    dataset_class = getattr(datasets, dataset_class_mapping[dataset])
    embedding = CNNLayerInversionCache.get_embedding_cache(dataset, str(layer_number))
    model, x_recon = CNNLayerInversionCache.get_model_cache(dataset, str(layer_number))
    recon_error = np.mean((dataset_class.get('data_numpy') - x_recon) ** 2, axis=1)

    fig = go.Figure()
    indices = dataset_class.get(key='data_indices')
    labels = dataset_class.get(key='data_labels')
    xy = np.stack([embedding[:, 0], embedding[:, 1]], axis=1)
    customdata = np.column_stack((indices, labels, xy))

    fig.add_trace(go.Scattergl(
        x=embedding[:, 0], y=embedding[:, 1], mode='markers',
        marker=dict(color=recon_error, colorscale='Inferno', size=2, colorbar=dict(title='Recon Error', thickness=10,
                                                                                   len=0.6, x=1.02)),
        name=f'CNN_Layer_{layer_number}',
        showlegend=False,
        customdata=customdata,
        hovertemplate='Latent X: %{customdata[2]:.2f}<br>Latent Y: %{customdata[3]:.2f}<br>Label: %{customdata[1]}'
    ))

    fig.update_layout(
        margin=dict(l=5, r=2, t=10, b=2),
        autosize=False,
        width=width,
        height=height,
    )
    return fig


def create_scatterplot(dataset, width, height, dashboard='euclidean', projection='TriMap', layer_number=0):
    if dashboard == 'euclidean':
        return dcc.Graph(
            figure=create_scatterplot_figure_euclidean(dataset, projection, width, height),
            id={'dashboard': dashboard, 'dataset': dataset, 'projection': projection, 'type': 'scatterplot'},
            className='plotly-plots',
            responsive=False,
            config={
                'displaylogo': False,
                'modeBarButtonsToRemove': ['autoscale'],
                'displayModeBar': True,
            }
        )
    elif dashboard == 'multi_recon':
        return dcc.Graph(
            figure=create_scatterplot_figure_multi_recon(dataset, projection, width, height),
            id={'dashboard': dashboard, 'dataset': dataset, 'projection': projection, 'type': 'scatterplot'},
            className='plotly-plots',
            responsive=False,
            config={
                'displaylogo': False,
                'modeBarButtonsToRemove': ['autoscale'],
                'displayModeBar': True,
            }
        )
    elif dashboard == 'model_prog':
        return dcc.Graph(
            figure=create_scatterplot_figure_model_prog(dataset, layer_number, width, height),
            id='model-prog-plot',
            className='plotly-plots',
            responsive=False,
            config={
                'displaylogo': False,
                'modeBarButtonsToRemove': ['autoscale'],
                'displayModeBar': True,
            }
        )
    else:
        print('Invalid Dashboard Passed')
