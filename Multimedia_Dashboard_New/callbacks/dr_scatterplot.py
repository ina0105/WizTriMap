from dash import ctx, Input, Output, State, MATCH
from dash.exceptions import PreventUpdate
import json
import threading

from app import app
from widgets import dr_scatterplot, recon_image, original_image
from helpers.utils import get_blank_image_base64
from helpers.config import model_prog_scatterplot_width, model_prog_scatterplot_height


euclidean_image_lock = threading.Lock()
multi_recon_image_lock = threading.Lock()
model_prog_image_lock = threading.Lock()
layer_number_change_lock = threading.Lock()


@app.callback(
    Output({'dashboard': MATCH, 'dataset': MATCH, 'projection': MATCH, 'type': 'scatterplot'}, 'figure'),
    Input({'dashboard': MATCH, 'dataset': MATCH, 'projection': MATCH, 'type': 'scatterplot'}, 'relayoutData'),
    State({'dashboard': MATCH, 'dataset': MATCH, 'projection': MATCH, 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def scatterplot_is_zoomed(zoom_data, scatterplot_fig):
    triggered_id = ctx.triggered_id
    dashboard = triggered_id['dashboard']
    dataset = triggered_id['dataset']
    method = triggered_id['projection']

    if len(zoom_data) == 1 and 'dragmode' in zoom_data:
        raise PreventUpdate

    if not any(key.startswith('xaxis.range') for key in zoom_data):
        raise PreventUpdate

    return dr_scatterplot.add_images_to_scatterplot(dataset, scatterplot_fig, dashboard)


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_mnist_trimap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'MNIST', 'TriMap')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'MNIST', scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'TriMap', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'TriMap', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'TriMap', 'type': 'scatterplot'},
          'figure'),
    prevent_initial_call=True
)
def euclidean_fashion_mnist_trimap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'FashionMNIST', 'TriMap')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'FashionMNIST',
                                                                                scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'TriMap', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'TriMap', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_cifar_trimap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'CIFAR-100', 'TriMap')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'CIFAR-100',
                                                                                scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-mnist-trimap-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_mnist_trimap_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'TriMap')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'TriMap', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input('euclidean-fashion-mnist-trimap-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'TriMap', 'type': 'scatterplot'},
          'figure'),
    prevent_initial_call=True
)
def euclidean_fashion_mnist_trimap_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'TriMap')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-cifar-trimap-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_cifar_trimap_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'TriMap')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'UMAP', 'type': 'scatterplot'}, 'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_mnist_umap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'MNIST', 'UMAP')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'MNIST', scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'UMAP', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'UMAP', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'UMAP', 'type': 'scatterplot'},
          'figure'),
    prevent_initial_call=True
)
def euclidean_fashion_mnist_umap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'FashionMNIST', 'UMAP')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'FashionMNIST',
                                                                                scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'UMAP', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_cifar_umap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'CIFAR-100', 'UMAP')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'CIFAR-100',
                                                                                scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-mnist-umap-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_mnist_umap_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'UMAP')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'UMAP', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input('euclidean-fashion-mnist-umap-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_fashion_mnist_umap_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'UMAP')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-cifar-umap-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'UMAP', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_cifar_umap_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'UMAP')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 't_SNE', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 't_SNE', 'type': 'scatterplot'}, 'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 't_SNE', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_mnist_tsne_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'MNIST', 't_SNE')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'MNIST', scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 't_SNE', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 't_SNE', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 't_SNE', 'type': 'scatterplot'},
          'figure'),
    prevent_initial_call=True
)
def euclidean_fashion_mnist_tsne_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'FashionMNIST', 't_SNE')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'FashionMNIST',
                                                                                scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 't_SNE', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 't_SNE', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 't_SNE', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_cifar_tsne_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'CIFAR-100', 't_SNE')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'CIFAR-100',
                                                                                scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 't_SNE', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-mnist-tsne-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 't_SNE', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_mnist_tsne_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 't_SNE')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 't_SNE', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input('euclidean-fashion-mnist-tsne-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 't_SNE', 'type': 'scatterplot'},
          'figure'),
    prevent_initial_call=True
)
def euclidean_fashion_mnist_tsne_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 't_SNE')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 't_SNE', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-cifar-tsne-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 't_SNE', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_cifar_tsne_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 't_SNE')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'PCA', 'type': 'scatterplot'}, 'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_mnist_pca_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'MNIST', 'PCA')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'MNIST', scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'PCA', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'PCA', 'type': 'scatterplot'},
          'figure'),
    prevent_initial_call=True
)
def euclidean_fashion_mnist_pca_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'FashionMNIST', 'PCA')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'FashionMNIST',
                                                                                scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'PCA', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_cifar_pca_scatterplot_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, 'CIFAR-100', 'PCA')
        original_image_object = original_image.display_euclidean_original_image(click_data, 'CIFAR-100',
                                                                                scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-mnist-pca-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'MNIST', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_mnist_pca_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'PCA')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-fashion-mnist-pca-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'FashionMNIST', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_fashion_mnist_pca_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'PCA')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-euclidean', 'src', allow_duplicate=True),
    Output('original-image-euclidean', 'src', allow_duplicate=True),
    Output({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure',
           allow_duplicate=True),
    Input('euclidean-cifar-pca-plot-click', 'data'),
    State({'dashboard': 'euclidean', 'dataset': 'CIFAR-100', 'projection': 'PCA', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def euclidean_cifar_pca_scatterplot_empty_area_clicked_display(click_data, scatterplot_fig):
    global euclidean_image_lock
    dataset = json.loads(click_data['graphId'])['dataset']
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with euclidean_image_lock:
        reconstructed_image_object = recon_image.display_euclidean_recon_image(click_data, dataset, 'PCA')
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('multi-recon-image-2d', 'src', allow_duplicate=True),
    Output('multi-recon-image-3d', 'src', allow_duplicate=True),
    Output('multi-recon-image-4d', 'src', allow_duplicate=True),
    Output('multi-recon-image-5d', 'src', allow_duplicate=True),
    Output('multi-recon-image-6d', 'src', allow_duplicate=True),
    Output('multi-recon-image-7d', 'src', allow_duplicate=True),
    Output('original-image-multi-recon', 'src', allow_duplicate=True),
    Output({'dashboard': 'multi_recon', 'dataset': 'MNIST', 'projection': 'TriMap', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input({'dashboard': 'multi_recon', 'dataset': 'MNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'clickData'),
    State({'dashboard': 'multi_recon', 'dataset': 'MNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def multi_recon_mnist_trimap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global multi_recon_image_lock
    with multi_recon_image_lock:
        reconstructed_image_object_list = recon_image.display_multi_recon_recon_image(click_data, 'MNIST', 'TriMap')
        original_image_object = original_image.display_multi_recon_original_image(click_data, 'MNIST', scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object_list[0], reconstructed_image_object_list[1], \
               reconstructed_image_object_list[2], reconstructed_image_object_list[3], \
               reconstructed_image_object_list[4], reconstructed_image_object_list[5], original_image_object, \
               scatterplot_fig


@app.callback(
    Output('multi-recon-image-2d', 'src', allow_duplicate=True),
    Output('multi-recon-image-3d', 'src', allow_duplicate=True),
    Output('multi-recon-image-4d', 'src', allow_duplicate=True),
    Output('multi-recon-image-5d', 'src', allow_duplicate=True),
    Output('multi-recon-image-6d', 'src', allow_duplicate=True),
    Output('multi-recon-image-7d', 'src', allow_duplicate=True),
    Output('original-image-multi-recon', 'src', allow_duplicate=True),
    Output({'dashboard': 'multi_recon', 'dataset': 'FashionMNIST', 'projection': 'TriMap', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input({'dashboard': 'multi_recon', 'dataset': 'FashionMNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'clickData'),
    State({'dashboard': 'multi_recon', 'dataset': 'FashionMNIST', 'projection': 'TriMap', 'type': 'scatterplot'}, 'figure'),
    prevent_initial_call=True
)
def multi_recon_fashion_mnist_trimap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global multi_recon_image_lock
    with multi_recon_image_lock:
        reconstructed_image_object_list = recon_image.display_multi_recon_recon_image(click_data, 'FashionMNIST',
                                                                                      'TriMap')
        original_image_object = original_image.display_multi_recon_original_image(click_data, 'FashionMNIST',
                                                                                  scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object_list[0], reconstructed_image_object_list[1], \
               reconstructed_image_object_list[2], reconstructed_image_object_list[3], \
               reconstructed_image_object_list[4], reconstructed_image_object_list[5], original_image_object, \
               scatterplot_fig


@app.callback(
    Output('multi-recon-image-2d', 'src', allow_duplicate=True),
    Output('multi-recon-image-3d', 'src', allow_duplicate=True),
    Output('multi-recon-image-4d', 'src', allow_duplicate=True),
    Output('multi-recon-image-5d', 'src', allow_duplicate=True),
    Output('multi-recon-image-6d', 'src', allow_duplicate=True),
    Output('multi-recon-image-7d', 'src', allow_duplicate=True),
    Output('original-image-multi-recon', 'src', allow_duplicate=True),
    Output({'dashboard': 'multi_recon', 'dataset': 'CIFAR-100', 'projection': 'TriMap', 'type': 'scatterplot'},
           'figure', allow_duplicate=True),
    Input({'dashboard': 'multi_recon', 'dataset': 'CIFAR-100', 'projection': 'TriMap', 'type': 'scatterplot'},
          'clickData'),
    State({'dashboard': 'multi_recon', 'dataset': 'CIFAR-100', 'projection': 'TriMap', 'type': 'scatterplot'},
          'figure'),
    prevent_initial_call=True
)
def multi_recon_cifar_trimap_scatterplot_clicked_display(click_data, scatterplot_fig):
    global multi_recon_image_lock
    with multi_recon_image_lock:
        reconstructed_image_object_list = recon_image.display_multi_recon_recon_image(click_data, 'CIFAR-100', 'TriMap')
        original_image_object = original_image.display_multi_recon_original_image(click_data, 'CIFAR-100',
                                                                                  scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object_list[0], reconstructed_image_object_list[1], \
               reconstructed_image_object_list[2], reconstructed_image_object_list[3], \
               reconstructed_image_object_list[4], reconstructed_image_object_list[5], original_image_object, \
               scatterplot_fig


@app.callback(
    Output('recon-image-model-prog', 'src', allow_duplicate=True),
    Output('original-image-model-prog', 'src', allow_duplicate=True),
    Output('model-prog-plot', 'figure', allow_duplicate=True),
    Input('model-prog-plot', 'clickData'),
    State('model-prog-plot', 'figure'),
    State('right-dropdown', 'value'),
    State('layer-selector-radio-buttons', 'value'),
    prevent_initial_call=True
)
def model_prog_trimap_scatterplot_clicked_display(click_data, scatterplot_fig, dataset, layer_number):
    global model_prog_image_lock
    with model_prog_image_lock:
        reconstructed_image_object = recon_image.display_cnn_layer_emb_recon_image(click_data, dataset, layer_number)
        original_image_object = original_image.display_cnn_layer_emb_original_image(click_data, dataset,
                                                                                    scatterplot_fig)
        clicked_x = click_data['points'][0]['x']
        clicked_y = click_data['points'][0]['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('recon-image-model-prog', 'src', allow_duplicate=True),
    Output('original-image-model-prog', 'src', allow_duplicate=True),
    Output('model-prog-plot', 'figure', allow_duplicate=True),
    Input('model-prog-trimap-plot-click', 'data'),
    State('right-dropdown', 'value'),
    State('layer-selector-radio-buttons', 'value'),
    State('model-prog-plot', 'figure'),
    prevent_initial_call=True
)
def model_prog_trimap_scatterplot_empty_area_clicked_display(click_data, dataset, layer_number, scatterplot_fig):
    global model_prog_image_lock
    click_data = {key: value for key, value in click_data.items() if key in ['x', 'y']}
    with model_prog_image_lock:
        reconstructed_image_object = recon_image.display_cnn_layer_emb_recon_image(click_data, dataset, layer_number)
        original_image_object = get_blank_image_base64()
        clicked_x = click_data['x']
        clicked_y = click_data['y']
        scatterplot_fig['data'][1]['x'] = [clicked_x]
        scatterplot_fig['data'][1]['y'] = [clicked_y]
        return reconstructed_image_object, original_image_object, scatterplot_fig


@app.callback(
    Output('model-prog-plot', 'figure', allow_duplicate=True),
    Input('model-prog-plot', 'relayoutData'),
    State('model-prog-plot', 'figure'),
    State('right-dropdown', 'value'),
    prevent_initial_call=True
)
def model_prog_scatterplot_is_zoomed(zoom_data, scatterplot_fig, dataset):
    if len(zoom_data) == 1 and 'dragmode' in zoom_data:
        raise PreventUpdate

    if not any(key.startswith('xaxis.range') for key in zoom_data):
        raise PreventUpdate

    return dr_scatterplot.add_images_to_scatterplot(dataset, scatterplot_fig, 'model_prog')


@app.callback(
    Output('model-prog-plot', 'figure', allow_duplicate=True),
    Input('layer-selector-radio-buttons', 'value'),
    State('right-dropdown', 'value'),
    prevent_initial_call=True
)
def model_prog_trimap_radio_button(layer_number, dataset):
    global layer_number_change_lock
    with layer_number_change_lock:
        return dr_scatterplot.create_scatterplot_figure_model_prog(dataset, layer_number, model_prog_scatterplot_width,
                                                                   model_prog_scatterplot_height)
