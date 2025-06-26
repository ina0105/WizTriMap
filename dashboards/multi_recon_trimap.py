from dash import html
import dash_bootstrap_components as dbc

from widgets import dr_scatterplot, recon_image, original_image
from helpers.config import multi_recon_scatterplot_width, multi_recon_scatterplot_height


def create_multi_recon_dashboard(dataset):
    scatterplot_trimap = dr_scatterplot.create_scatterplot(dataset, multi_recon_scatterplot_width,
                                                           multi_recon_scatterplot_height, dashboard='multi_recon')
    original_image_widget = original_image.create_multi_recon_original_image_widget()
    multi_recon_image_grid_widget = recon_image.create_multi_recon_image_grid()
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div("TriMap", className='panel-label'),
                scatterplot_trimap],
                className='multi-recon-top-row-col-left border-widget', style={'width': '84%'}),
            dbc.Col([
                    html.Div("Original Image", className='panel-label'),
                    original_image_widget],
                    className='multi-recon-top-row-col-right border-widget', style={'width': '16%'})
        ], className='multi-recon-trimap-top-row', justify='between'),
        multi_recon_image_grid_widget
    ], fluid=True, id='multi-recon-trimap-container')
