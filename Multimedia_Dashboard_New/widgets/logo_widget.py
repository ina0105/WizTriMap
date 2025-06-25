from dash import html
from PIL import Image

from helpers.utils import pil_to_base64
from helpers.config import logo_image_resize_value


def get_logo_image():
    logo_image = pil_to_base64(Image.open('logo_image/WizTriMap.png').resize(logo_image_resize_value,
                                                                             resample=Image.NEAREST))
    return logo_image


def create_logo_image_widget():
    return html.Div([html.Img(id='logo-image', src=get_logo_image())], id='logo-image-div')
