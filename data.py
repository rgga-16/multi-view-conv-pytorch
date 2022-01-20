import torch 
from torch.utils.data import Dataset
from torchvision import transforms as T
import os, random

from utils import configs, device, load_image,itot,toti,extract_bool_mask,bool_to_real_mask
import args
from args import args_

class Data(Dataset):
    def __init__(self,root,set,category,views,shape_list,shuffle=True):

        self.sketch_model_pairs = []
        self.set=set
        self.category = category

        self.n_source_views = len(args_.sketch_views)
        # self.n_dnfs_views = max(2,n_source_views)
        self.n_dnfs_views=args_.num_dnfs_views
        self.n_dn_views = args_.num_dn_views
        self.n_target_views = self.n_dnfs_views + self.n_dn_views

        self.shape_list = shape_list 
        # self.dn_lists = []
        # self.dnfs_lists = []
        # self.mask_list = []
        # self.target_lists = []
        # self.sketch_lists = []
       
        self.load_files(root,shape_list)

        # assert len(self.dn_lists)==len(self.dnfs_lists)
        # assert len(self.dn_lists)==len(self.sketch_lists)
        # assert len(self.dnfs_lists)==len(self.sketch_lists)
        

    
    def load_files(self,root,shape_list):
        imsize = args_.imsize

        for shape_name in shape_list:
            
            dnfs_files = [os.path.join(root,'dnfs',shape_name,f'dnfs-256-{dnfs_view}.png') for dnfs_view in range(self.n_dnfs_views)]
            dn_files = [os.path.join(root,'dn',shape_name,f'dn-256-{dn_view}.png') for dn_view in range(self.n_dn_views)]
            
            if self.set=='test':
                variation=0
            else:
                variation = random.randint(0,args_.sketch_variations-1)
            sketch_files = [os.path.join(root,'sketch',shape_name,f'sketch-{view}-{variation}.png') for view in args_.sketch_views]
            # sketch_files = [os.path.join(root,'sketch',shape_name,f'sketch-{view}-{var}.png') for view in args_.sketch_views for var in range(args_.sketch_variations)]

            shape = {}
            shape['name'] = shape_name 
            # shape['targets'] = targets_list 
            # shape['sketches'] = sketch_list 
            # shape['dn'] = dn_list 
            # shape['dnfs'] = dnfs_list
            # shape['mask'] = real_mask

            shape['dn_f'] = dn_files 
            shape['dnfs_f'] = dnfs_files 
            shape['sketches_f'] = sketch_files
            self.sketch_model_pairs.append(shape)

            # self.dn_lists.append(dn_list)
            # self.dnfs_lists.append(dnfs_list)
            # self.mask_list.append(real_mask)
            # self.sketch_lists.append(sketch_list)
            # self.target_lists.append(targets_list)

        return

    def __len__(self):
        return len(self.sketch_model_pairs)
    
    def __getitem__(self,index):
        shape = self.sketch_model_pairs[index]
        imsize = args_.imsize

        dn_list = torch.stack([itot(load_image(dn_f,'RGBA'),size=imsize) for dn_f in shape['dn_f']],dim=0)
        dnfs_list = torch.stack([itot(load_image(dnfs_f,'RGBA'),size=imsize) for dnfs_f in shape['dnfs_f']],dim=0)
        targets_list = torch.cat([dnfs_list,dn_list],dim=0)
        bool_masks = extract_bool_mask(targets_list)

        if configs['PREDICT_NORMAL']:
            b,c,h,w = targets_list.shape 
            target_background = torch.cat([torch.zeros((b,2,h,w),device=device),torch.ones((b,2,h,w),device=device)],dim=1)
            tiled = torch.tile(bool_masks,[1,c,1,1])
            targets_list = torch.where(tiled,targets_list,target_background)   
        else:
            # retain depth only
            targets_list = targets_list[:,3:,:,:]
        
        real_mask = bool_to_real_mask(bool_masks)
        targets_list = torch.cat([targets_list,real_mask],dim=1)

        sketch_list_init = torch.stack([itot(load_image(sketch_f,'L'),size=imsize) for sketch_f in shape['sketches_f']],dim=1).squeeze(0)
        sketch_list_flipped = torch.flip(sketch_list_init,[1,2])
        sketch_list = torch.cat([sketch_list_init,sketch_list_flipped],dim=0)


        return sketch_list, targets_list

def load_train_data(root,category,views):

    shape_list_file = open(os.path.join(root,'train-list.txt'))
    shape_list = shape_list_file.read().splitlines()
    shape_list_file.close() 

    assert category in ['Airplane','Chair','Character']

    return Data(root,'train',category,views,shape_list)

def load_val_data(root,category,views):

    shape_list_file = open(os.path.join(root,'validate-list.txt'))
    shape_list = shape_list_file.read().splitlines()
    shape_list_file.close() 

    assert category in ['Airplane','Chair','Character']

    return Data(root,'val',category,views,shape_list)


def load_test_data(root,category,style,views):

    assert style=='Draw' or style=='Synthetic'

    shape_list_file = None 

    shape_list_file = open(os.path.join(root,'test-list.txt'))
    shape_list = shape_list_file.read().splitlines()
    shape_list_file.close() 

    return Data(root,'test',category,views,shape_list)


def load_encode_data(root,category,views):
    return 

if __name__=='__main__':

    root = os.path.join(configs['TRAIN_DIR'],configs['CATEGORY'])
    
    train_data = load_train_data(root,configs['CATEGORY'],None)

    sketches,targets = train_data.__getitem__(0)
    