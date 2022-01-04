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

'''
Converts PIL Image to Torch tensor
'''
def itot(image):
    transform = T.Compose([
        T.Resize(configs['IMSIZE']),
        T.ToTensor()
    ])

    tensor = transform(image)

    return tensor 

'''
Converts tensor to PIL Image
'''
def toti(tensor):

    transform = T.Compose([
        T.ToPILImage()
    ])

    image = transform(tensor)

    return image


global configs 
configs = read_configs()