import yaml 
import torch,torchvision

from matplotlib import pyplot as plt
import os,utils 
from PIL import Image

from torchvision import transforms as T

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_configs(path='./config.yaml'):
    with open(path) as f:
        configs = yaml.load(f,Loader=yaml.FullLoader)
        f.close()
    return configs

def load_image(filename,mode='RGB'):
    img = Image.open(filename).convert(mode)
    return img

def denormalize(tensor):

    mean = (0.5,0.5,0.5) if tensor.shape[0]==3 else (0.5)
    std = (0.5,0.5,0.5) if tensor.shape[0]==3 else (0.5)
    denormalized = tensor * std + mean
    return denormalized

'''
Converts PIL Image to Torch tensor
'''
def itot(image,size=None):

    if size:
        resizer = T.Resize(size)
        image = resizer(image)
    
    transform = T.Compose([
        T.Resize(size),
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

def real_to_bool_mask(real_mask):
    bool_mask=torch.where(torch.greater(real_mask,torch.zeros_like(real_mask,device=device)),
                                        torch.ones_like(real_mask,dtype=torch.bool,device=device),
                                        torch.zeros_like(real_mask,dtype=torch.bool,device=device))
    return bool_mask 

def bool_to_real_mask(bool_mask):
    ones = torch.ones_like(bool_mask,dtype=torch.float32)
    negatives = torch.negative(ones)
    real_mask = torch.where(bool_mask,ones,negatives)
    return real_mask

def apply_mask(content,mask):
    c = content.shape[1]
    if c > 1:
        mask = torch.tile(mask,(1,c,1,1))
    masked = torch.where(torch.greater(mask,0),content,torch.ones_like(content,device=device))
    return masked

def show_images(images,normalize=True):
    img_grid = torchvision.utils.make_grid(images,normalize=normalize)
    plt.axis('off')
    plt.imshow(img_grid.cpu().permute(1,2,0))
    plt.show()

    
    plt.clf()

global configs 
configs = read_configs()

global device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")