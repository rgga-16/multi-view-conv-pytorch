import torch 
from torch.utils.data import Dataset
import os 

from utils import configs

class Data(Dataset):
    def __init__(self,data_dir,shuffle=True):
        shape_list_f = open(os.path.join(data_dir,'train-list.txt'))
        shape_list = shape_list_f.read().splitlines()
        shape_list_f.close()




        print()
        
    def __len__(self):

        return 
    
    def __getitem__(self,index):

        return 


if __name__=='__main__':

    train_data = Data('./data/train/Airplane')
    print()