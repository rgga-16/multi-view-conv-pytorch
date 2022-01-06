import yaml 
import torch
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
        T.ToTensor(),
    ])

    tensor = transform(image)

    mean = (0.5,0.5,0.5) if tensor.shape[0]==3 else (0.5)
    std = (0.5,0.5,0.5) if tensor.shape[0]==3 else (0.5)
    tensor = T.Normalize(mean,std)(tensor)

    return tensor.to(device) 

'''
Converts tensor to PIL Image
'''
def toti(tensor):

    transform = T.Compose([
        T.ToPILImage()
    ])

    image = transform(tensor)

    return image

def extract_bool_mask(image):
    im_shape = image.shape
    assert im_shape[1]==4

    depth = image[:,3,:,:].unsqueeze(1)
    trues = torch.ones_like(depth,dtype=torch.bool)
    falses = torch.zeros_like(depth,dtype=torch.bool)
    bool_mask = torch.where(depth<0.9,trues,falses)

    return bool_mask


global configs 
configs = read_configs()

global device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")