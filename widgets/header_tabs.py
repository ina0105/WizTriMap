from dash import dcc


def create_header_tabs():
    return dcc.Tabs(id='main-tabs',
                    value='euclidean',
                    children=[
                        dcc.Tab(label='Method Comparison', value='euclidean', className='tab-item'),
                        dcc.Tab(label='Multi-Dimensional Reconstruction', value='multi_recon', className='tab-item'),
                        dcc.Tab(label='CNN Layer Visualization', value='model_prog', className='tab-item')],
                    className='custom-tabs')
