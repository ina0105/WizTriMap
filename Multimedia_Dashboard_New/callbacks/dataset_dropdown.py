from dash import html, Input, Output, State
from dash.exceptions import PreventUpdate
import time

from app import app
from dashboards import euclidean, multi_recon_trimap, model_prog
from helpers.config import dataset_list


@app.callback(
    Output('selected-tab', 'data', allow_duplicate=True),
    Input('right-dropdown', 'value'),
    State('main-tabs', 'value'),
    prevent_initial_call=True
)
def store_tab_selection(dataset_dropdown_value, tab_value):
    if dataset_dropdown_value not in dataset_list:
        raise PreventUpdate
    if tab_value == 'model_prog':
        time.sleep(6)
    else:
        time.sleep(3)
    return ''


@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('right-dropdown', 'value'),
    State('main-tabs', 'value'),
    prevent_initial_call=True
)
def render_dashboard_from_dataset_dropdown(dataset_dropdown_value, tab_state):
    if dataset_dropdown_value not in dataset_list:
        raise PreventUpdate
    if tab_state == 'euclidean':
        print(f'Rendering {dataset_dropdown_value} {tab_state.capitalize()} Dashboard')
        return euclidean.create_euclidean_dashboard(dataset_dropdown_value)
    elif tab_state == 'multi_recon':
        print(f'Rendering {dataset_dropdown_value} {tab_state.capitalize()} Dashboard')
        return multi_recon_trimap.create_multi_recon_dashboard(dataset_dropdown_value)
    elif tab_state == 'model_prog':
        print(f'Rendering {dataset_dropdown_value} {tab_state.capitalize()} Dashboard')
        return model_prog.create_model_prog_dashboard(dataset_dropdown_value)
    return html.Div("Invalid Tab")
