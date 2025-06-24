import base64
import io
from PIL import Image


def pil_to_base64(pil_image):
    buff = io.BytesIO()
    pil_image.save(buff, format='png')
    base64_img = base64.b64encode(buff.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{base64_img}'


def get_blank_image_base64():
    white_image = Image.new('RGB', (150, 150), (255, 255, 255))
    return pil_to_base64(white_image)
