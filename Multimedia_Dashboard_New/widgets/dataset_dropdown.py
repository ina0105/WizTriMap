from dash import dcc, html


def create_dataset_dropdown():
    return html.Div(
               dcc.Dropdown(
                   id='right-dropdown',
                   options=[
                       {'label': 'MNIST', 'value': 'MNIST'},
                       {'label': 'FashionMNIST', 'value': 'FashionMNIST'},
                       {'label': 'CIFAR-100', 'value': 'CIFAR-100'}
                   ],
                   value='MNIST',
                   style={'width': '200px'}
               ),
               style={'marginLeft': '770.5px', 'position': 'relative', 'zindex': 1}
           )
