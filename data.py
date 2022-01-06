import torch 
from torch.utils.data import Dataset
from torchvision import transforms as T
import os, random

from utils import configs, load_image,itot,toti,extract_bool_mask,bool_to_real_mask

class Data(Dataset):
    def __init__(self,root,set,category,views,shape_list,shuffle=True,batch_size=-1):

        self.sketch_model_pairs = []
        self.set=set
        self.category = category

        self.n_source_views = len(configs['SKETCH_VIEWS'])
        # self.n_dnfs_views = max(2,n_source_views)
        self.n_dnfs_views=2
        self.n_dn_views = configs['NUM_DN_VIEWS']
        self.n_target_views = self.n_dnfs_views + self.n_dn_views

        self.shape_list = shape_list 
        self.dn_lists = []
        self.dnfs_lists = []
        self.target_lists = []
        self.sketch_lists = []
       
        self.load_files(root,shape_list)

        assert len(self.dn_lists)==len(self.dnfs_lists)
        assert len(self.dn_lists)==len(self.sketch_lists)
        assert len(self.dnfs_lists)==len(self.sketch_lists)
        print()

    
    def load_files(self,root,shape_list):
        imsize = configs['IMSIZE']

        for shape_name in shape_list:
            dn_files = [os.path.join(root,'dn',shape_name,f'dn-{imsize}-{dn_view}.png') for dn_view in range(self.n_dn_views)]
            dnfs_files = [os.path.join(root,'dnfs',shape_name,f'dnfs-{imsize}-{dnfs_view}.png') for dnfs_view in range(self.n_dnfs_views)]
            
            if self.set=='test':
                variation=0
            else:
                variation = random.randint(0,configs['SKETCH_VARIATIONS']-1)
            sketch_files = [os.path.join(root,'sketch',shape_name,f'sketch-{view}-{variation}.png') for view in configs['SKETCH_VIEWS']]
            # sketch_files = [os.path.join(root,'sketch',shape_name,f'sketch-{view}-{var}.png') for view in configs['SKETCH_VIEWS'] for var in range(configs['SKETCH_VARIATIONS'])]
            
            dn_list = torch.stack([itot(load_image(dn_f,'RGBA')) for dn_f in dn_files],dim=0)
            dnfs_list = torch.stack([itot(load_image(dnfs_f,'RGBA')) for dnfs_f in dnfs_files],dim=0)
            targets_list = torch.cat([dnfs_list,dn_list],dim=0)
            bool_masks = extract_bool_mask(targets_list)

            real_masks = bool_to_real_mask(bool_masks)
            targets = torch.cat([targets_list,bool_masks],dim=1)

            sketch_list_init = torch.stack([itot(load_image(sketch_f,'L')) for sketch_f in sketch_files],dim=1).squeeze(0)
            sketch_list_flipped = torch.flip(sketch_list_init,[1,2])
            sketch_list = torch.cat([sketch_list_init,sketch_list_flipped],dim=0)

            shape = {}
            shape['name'] = shape_name 
            shape['targets'] = targets_list 
            shape['sketches'] = sketch_list 
            shape['dn'] = dn_list 
            shape['dnfs'] = dnfs_list
            self.sketch_model_pairs.append(shape)

            self.dn_lists.append(dn_list)
            self.dnfs_lists.append(dnfs_list)
            self.sketch_lists.append(sketch_list)
            self.target_lists.append(targets_list)

        return
    

    def __len__(self):
        return len(self.sketch_lists)
    
    def __getitem__(self,index):
        shape = self.sketch_model_pairs[index]


        return shape['sketches'], shape['targets']

def load_train_data(root,category,views,batch_size=-1):

    shape_list_file = open(os.path.join(root,'train-list.txt'))
    shape_list = shape_list_file.read().splitlines()
    shape_list_file.close() 

    assert category in ['Airplane','Chair','Character']

    return Data(root,'train',category,views,shape_list,batch_size=batch_size)

def load_val_data(root,category,views,batch_size=-1):

    shape_list_file = open(os.path.join(root,'validate-list.txt'))
    shape_list = shape_list_file.read().splitlines()
    shape_list_file.close() 

    assert category in ['Airplane','Chair','Character']

    return Data(root,'val',category,views,shape_list,batch_size=batch_size)


def load_test_data(root,category,style,views,batch_size=-1):

    assert style=='Draw' or style=='Synthetic'

    shape_list_file = None 

    shape_list_file = open(os.path.join(root,'test-list.txt'))
    shape_list = shape_list_file.read().splitlines()
    shape_list_file.close() 

    return Data(root,'test',category,views,shape_list,batch_size=batch_size)


def load_encode_data(root,category,views,batch_size=-1):
    return 

if __name__=='__main__':

    root = os.path.join(configs['TRAIN_DIR'],configs['CATEGORY'])
    
    train_data = load_train_data(root,configs['CATEGORY'],None)

    sketches,targets = train_data.__getitem__(0)
    print()