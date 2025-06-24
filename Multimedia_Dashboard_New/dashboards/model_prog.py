from dash import html, dcc
import dash_bootstrap_components as dbc

from widgets import dr_scatterplot, recon_image, original_image, radio_button
from helpers.config import model_prog_scatterplot_width, model_prog_scatterplot_height


def create_model_prog_dashboard(dataset):
    scatterplot_trimap = dr_scatterplot.create_scatterplot(dataset, model_prog_scatterplot_width,
                                                           model_prog_scatterplot_height, dashboard='model_prog')
    original_image_widget = original_image.create_model_prog_original_image_widget()
    recon_image_widget = recon_image.create_model_prog_recon_image_widget()
    radio_button_widget = radio_button.create_radio_button_widget()
    return html.Div([
            radio_button_widget,
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div("TriMap", className='panel-label'),
                        dcc.Loading(id='loading-original-image-euclidean', children=[
                            scatterplot_trimap], type='circle')
                    ],
                        className='model-prog-top-row-col-left border-widget', style={'width': '84%'}),
                    dbc.Col([
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Original Image", className='panel-label'),
                                    original_image_widget
                                ], className='model-prog-top-row-col-right-sub-row-top-column',
                                    style={'width': '100%'})
                            ], className='model-prog-top-row-col-right-sub-row-top', style={'height': '100%'})
                        ], className='model-prog-top-row-col-right-sub-row-top-div', style={'height': '49%'}),
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Div("Reconstructed Image", className='panel-label-bottom'),
                                    recon_image_widget
                                ], className='model-prog-top-row-col-right-sub-row-bottom-column',
                                    style={'width': '100%'})
                            ], className='model-prog-top-row-col-right-sub-row-bottom', style={'height': '100%'})
                        ], className='model-prog-top-row-col-right-sub-row-bottom-div', style={'height': '49%'})
                    ], className='model-prog-top-row-col-right border-widget', style={'width': '16%'})
                ], className='model-prog-trimap-top-row', justify='between')
            ], fluid=True, id='model-prog-trimap-container')
        ])
