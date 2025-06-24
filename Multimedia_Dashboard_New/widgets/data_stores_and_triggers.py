from dash import dcc, html


def create_data_stores_and_triggers():
    return html.Div([
        dcc.Store(id='euclidean-mnist-trimap-plot-click', data={}),
        dcc.Store(id='euclidean-fashion-mnist-trimap-plot-click', data={}),
        dcc.Store(id='euclidean-cifar-trimap-plot-click', data={}),
        dcc.Store(id='euclidean-mnist-umap-plot-click', data={}),
        dcc.Store(id='euclidean-fashion-mnist-umap-plot-click', data={}),
        dcc.Store(id='euclidean-cifar-umap-plot-click', data={}),
        dcc.Store(id='euclidean-mnist-tsne-plot-click', data={}),
        dcc.Store(id='euclidean-fashion-mnist-tsne-plot-click', data={}),
        dcc.Store(id='euclidean-cifar-tsne-plot-click', data={}),
        dcc.Store(id='euclidean-mnist-pca-plot-click', data={}),
        dcc.Store(id='euclidean-fashion-mnist-pca-plot-click', data={}),
        dcc.Store(id='euclidean-cifar-pca-plot-click', data={}),
        dcc.Store(id='model-prog-trimap-plot-click', data={}),
        html.Button(id="euclidean-mnist-trimap-trigger", style={"display": "none"}),
        html.Button(id="euclidean-fashion-mnist-trimap-trigger", style={"display": "none"}),
        html.Button(id="euclidean-cifar-trimap-trigger", style={"display": "none"}),
        html.Button(id="euclidean-mnist-umap-trigger", style={"display": "none"}),
        html.Button(id="euclidean-fashion-mnist-umap-trigger", style={"display": "none"}),
        html.Button(id="euclidean-cifar-umap-trigger", style={"display": "none"}),
        html.Button(id="euclidean-mnist-tsne-trigger", style={"display": "none"}),
        html.Button(id="euclidean-fashion-mnist-tsne-trigger", style={"display": "none"}),
        html.Button(id="euclidean-cifar-tsne-trigger", style={"display": "none"}),
        html.Button(id="euclidean-mnist-pca-trigger", style={"display": "none"}),
        html.Button(id="euclidean-fashion-mnist-pca-trigger", style={"display": "none"}),
        html.Button(id="euclidean-cifar-pca-trigger", style={"display": "none"}),
        html.Button(id="model-prog-trimap-trigger", style={"display": "none"})
    ], id="datastore_and_trigger_div")
