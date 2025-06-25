from dash import html
import dash_bootstrap_components as dbc

from widgets import dr_scatterplot, recon_image, original_image
from helpers.config import euclidean_scatterplot_width, euclidean_scatterplot_height


def create_euclidean_dashboard(dataset):
    scatterplot_pca = dr_scatterplot.create_scatterplot(dataset, euclidean_scatterplot_width,
                                                        euclidean_scatterplot_height, projection='PCA')
    scatterplot_t_sne = dr_scatterplot.create_scatterplot(dataset, euclidean_scatterplot_width,
                                                          euclidean_scatterplot_height, projection='t_SNE')
    scatterplot_umap = dr_scatterplot.create_scatterplot(dataset, euclidean_scatterplot_width,
                                                         euclidean_scatterplot_height, projection='UMAP')
    scatterplot_trimap = dr_scatterplot.create_scatterplot(dataset, euclidean_scatterplot_width,
                                                           euclidean_scatterplot_height)
    recon_image_widget = recon_image.create_euclidean_recon_image_widget()
    original_image_widget = original_image.create_euclidean_original_image_widget()
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div("TriMap", className='panel-label'),
                scatterplot_trimap],
                className='main-col-left border-widget', style={'width': '42%'}),
            dbc.Col([
                html.Div("Original Image", className='panel-label'),
                original_image_widget], 
                className='main-col-mid border-widget', style={'width': '16%'}),
            dbc.Col([
                html.Div("UMAP", className='panel-label'), 
                scatterplot_umap],
                className='main-col-right border-widget', style={'width': '42%'})
        ], className='top-row', justify='between'),
        dbc.Row([
            dbc.Col([
                html.Div("T-SNE", className='panel-label'), 
                scatterplot_t_sne],
                className='main-col-left border-widget', style={'width': '42%'}),
            dbc.Col([html.Div("Reconstructed Image", className='panel-label'), 
                     recon_image_widget], 
                    className='main-col-mid border-widget', style={'width': '16%'}),
            dbc.Col([html.Div("PCA", className='panel-label'), 
                     scatterplot_pca],
                    className='main-col-right border-widget', style={'width': '42%'})
        ], className='bottom-row', justify='between')
    ], fluid=True, id='euclidean-container')
