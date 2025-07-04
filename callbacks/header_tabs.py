from dash import html, Input, Output, State
import time

from app import app
from dashboards import euclidean, multi_recon_trimap, model_prog


@app.callback(
    Output('selected-tab', 'data', allow_duplicate=True),
    Input('init-load', 'data'),
    Input('main-tabs', 'value'),
    prevent_initial_call=True
)
def store_tab_selection(init_load_flag, tab_value):
    time.sleep(3)
    return ''


@app.callback(
    Output('main-content', 'children', allow_duplicate=True),
    Input('init-load', 'data'),
    Input('main-tabs', 'value'),
    State('right-dropdown', 'value'),
    prevent_initial_call=True
)
def render_dashboard_from_header_tab(init_load_flag, tab_value, dataset_dropdown_state):
    if tab_value == 'euclidean':
        return euclidean.create_euclidean_dashboard(dataset_dropdown_state)
    elif tab_value == 'multi_recon':
        return multi_recon_trimap.create_multi_recon_dashboard(dataset_dropdown_state)
    elif tab_value == 'model_prog':
        return model_prog.create_model_prog_dashboard(dataset_dropdown_state)
    return html.Div("Invalid Tab")
