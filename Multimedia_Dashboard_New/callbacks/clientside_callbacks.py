from dash import Input, Output

from app import app

app.clientside_callback(
    """
    function(tab_value) {
        return true;
    }
    """,
    Output('init-load', 'data'),
    Input('init-trigger', 'n_intervals'),
    prevent_initial_call=False
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"MNIST","projection":"TriMap","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-mnist-trimap-plot-click', 'data'),
    Input('euclidean-mnist-trimap-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"FashionMNIST","projection":"TriMap","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-fashion-mnist-trimap-plot-click', 'data'),
    Input('euclidean-fashion-mnist-trimap-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"CIFAR-100","projection":"TriMap","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-cifar-trimap-plot-click', 'data'),
    Input('euclidean-cifar-trimap-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"MNIST","projection":"UMAP","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-mnist-umap-plot-click', 'data'),
    Input('euclidean-mnist-umap-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"FashionMNIST","projection":"UMAP","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-fashion-mnist-umap-plot-click', 'data'),
    Input('euclidean-fashion-mnist-umap-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"CIFAR-100","projection":"UMAP","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-cifar-umap-plot-click', 'data'),
    Input('euclidean-cifar-umap-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"MNIST","projection":"t_SNE","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-mnist-tsne-plot-click', 'data'),
    Input('euclidean-mnist-tsne-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"FashionMNIST","projection":"t_SNE","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-fashion-mnist-tsne-plot-click', 'data'),
    Input('euclidean-fashion-mnist-tsne-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"CIFAR-100","projection":"t_SNE","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-cifar-tsne-plot-click', 'data'),
    Input('euclidean-cifar-tsne-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"MNIST","projection":"PCA","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-mnist-pca-plot-click', 'data'),
    Input('euclidean-mnist-pca-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"FashionMNIST","projection":"PCA","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-fashion-mnist-pca-plot-click', 'data'),
    Input('euclidean-fashion-mnist-pca-trigger', 'n_clicks'),
    prevent_initial_call=True
)

app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = '{"dashboard":"euclidean","dataset":"CIFAR-100","projection":"PCA","type":"scatterplot"}'
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData, graphId: graphContainerId};
    }
    """,
    Output('euclidean-cifar-pca-plot-click', 'data'),
    Input('euclidean-cifar-pca-trigger', 'n_clicks'),
    prevent_initial_call=True
)


app.clientside_callback(
    """
    function(n_clicks) {
        const graphContainerId = "model-prog-plot"
        const graphContainer = document.getElementById(graphContainerId);
        const graph = graphContainer.querySelector('.js-plotly-plot');

        const bbox = graph.getBoundingClientRect();
        const layout = graph._fullLayout;
        const xaxis = layout.xaxis;
        const yaxis = layout.yaxis;

        const xPixel = window.lastClickEvent.clientX - bbox.left;
        const yPixel = window.lastClickEvent.clientY - bbox.top;

        const xData = xaxis.p2d(xPixel - xaxis._offset);
        const yData = yaxis.p2d(yPixel - yaxis._offset);

        return {x: xData, y: yData};
    }
    """,
    Output('model-prog-trimap-plot-click', 'data'),
    Input('model-prog-trimap-trigger', 'n_clicks'),
    prevent_initial_call=True
)
