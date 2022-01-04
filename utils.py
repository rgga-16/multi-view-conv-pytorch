import yaml 
import os 
from PIL import Image

from torchvision import transforms as T

def read_configs(path='./config.yaml'):
    with open(path) as f:
        configs = yaml.load(f,Loader=yaml.FullLoader)
        f.close()
    return configs

def load_image(filename,mode='RGB'):
    img = Image.open(filename).convert(mode)
    return img

def image_to_tensor(image):
    transform = T.Compose([
        T.Resize(configs['IMSIZE']),
        T.ToTensor()
    ])

    tensor = transform(image)

    return tensor 


global configs 
configs = read_configs()