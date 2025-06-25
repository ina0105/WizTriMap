from dash import html, dcc
import dash_bootstrap_components as dbc


def create_radio_button_widget():
    return html.Div([
        html.Div([
            html.Div("Layer Selector", className='panel-label-layer-selector'),
            dbc.RadioItems(
                options=[{"label": x, "value": x} for x in ['1', '2', '3']],
                value='1',
                inline=True,
                id='layer-selector-radio-buttons',
                class_name='layer-selector-radio-buttons')
        ], className='model-prog-layer-selector-div')
    ], className='model-prog-layer-selector-outer-div')
