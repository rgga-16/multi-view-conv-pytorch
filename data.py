import torch 
from torch.utils.data import Dataset
import os 

from utils import configs, load_image,image_to_tensor

class Data(Dataset):
    def __init__(self,root,category,views,shape_list,shapenet_ids=[],shuffle=True,batch_size=-1):

        self.sketch_model_pairs = []
        self.shapenet_ids = shapenet_ids

        n_source_views = len(configs['SKETCH_VIEWS'])
        # n_dnfs_views = max(2,n_source_views)
        n_dnfs_views=2
        n_dn_views = 12
        n_target_views = n_dnfs_views + n_dn_views
        imsize = configs['IMSIZE']

        self.dn_lists = []
        self.dnfs_lists = []
        self.sketch_lists = []

        if shapenet_ids:
            for snid in shapenet_ids:
                self.load_files_snid(root,shape_list,snid,n_dn_views,n_dnfs_views)  
        else:
            self.load_files(root,shape_list,n_dn_views,n_dnfs_views)
            #^Error. sketch_list, dn_list and dnfs_list are not appending their respective returned lists.
 
        print()

        # src_prefix_list = ['sketch/' for i in range(n_sketch_views)]

        # if configs['TEST']:
        #     sketch_variation = 0 
        # else:
        #     sketch_variations = [n for n in range(configs['SKETCH_VARIATIONS'])]

        # src_suffix_list = [f'-']
    
    def load_files(self,root,shape_list,n_dn_views,n_dnfs_views):
        imsize = configs['IMSIZE']

        for shape_name in shape_list:
            shape = {}
            shape['name'] = shape_name 

            dn_files = [os.path.join(root,'dn',shape_name,f'dn-{imsize}-{dn_view}.png') for dn_view in range(n_dn_views)]
            dnfs_files = [os.path.join(root,'dnfs',shape_name,f'dnfs-{imsize}-{dnfs_view}.png') for dnfs_view in range(n_dnfs_views)]
            sketch_files = [os.path.join(root,'sketch',shape_name,f'sketch-{view}-{var}.png') for view in configs['SKETCH_VIEWS'] for var in range(configs['SKETCH_VARIATIONS'])]
            
            dn_list = [image_to_tensor(load_image(dn_f)) for dn_f in dn_files]
            dnfs_list = [image_to_tensor(load_image(dnfs_f)) for dnfs_f in dnfs_files]
            sketch_list = [image_to_tensor(load_image(sketch_f,'L')) for sketch_f in sketch_files]
        
            self.dn_lists.append(dn_list)
            self.dnfs_lists.append(dnfs_list)
            self.sketch_lists.append(sketch_list)
        return
    
    def load_files_snid(self,root,shape_list,snid,n_dn_views,n_dnfs_views):
        imsize = configs['IMSIZE']

        for shape_name in shape_list:
            shape = {}
            shape['name'] = shape_name 

            dn_files = [os.path.join(root,'dn',shape_name,snid,f'dn-{imsize}-{dn_view}.png') for dn_view in range(n_dn_views)]
            dnfs_files = [os.path.join(root,'dnfs',shape_name,snid,f'dnfs-{imsize}-{dnfs_view}.png') for dnfs_view in range(n_dnfs_views)]
            sketch_files = [os.path.join(root,'sketch',shape_name,snid,f'sketch-{view}-{var}.png') for view in configs['SKETCH_VIEWS'] for var in range(configs['SKETCH_VARIATIONS'])]
            
            dn_list = [image_to_tensor(load_image(dn_f)) for dn_f in dn_files]
            dnfs_list = [image_to_tensor(load_image(dnfs_f)) for dnfs_f in dnfs_files]
            sketch_list = [image_to_tensor(load_image(sketch_f,'L')) for sketch_f in sketch_files]

            self.dn_lists.append(dn_list)
            self.dnfs_lists.append(dnfs_list)
            self.sketch_lists.append(sketch_list)
            
        return

    def __len__(self):

        return 
    
    def __getitem__(self,index):

        return 

def load_train_data(root,category,views,batch_size=-1):

    shape_list_file = open(os.path.join(root,'train-list.txt'))
    shape_list = shape_list_file.read().splitlines()
    shape_list_file.close() 

    if category=='Airplane':
        shapenet_ids = ['02691156']
    elif category=='Chair':
        shapenet_ids = ['03001627','04256520']
    elif category=='Character':
        shapenet_ids=[]
    else:
        shapenet_ids=[]
        # raise ValueError('Category should be either Airplane, Character, or Chair (case-sensitive).')
    return Data(root,category,views,shape_list,shapenet_ids,batch_size=batch_size)

def load_val_data(root,category,views,batch_size=-1):
    return 

def load_test_data(root,category,style,views,batch_size=-1):

    assert style=='Draw' or style=='Synthetic'

    shape_list_file = None 


    # shape_list_file = open(os.path.join(root,'train-list.txt'))
    # shape_list = shape_list_file.read().splitlines()
    # shape_list_file.close() 

    if category=='Airplane':
        shapenet_ids = ['02691156']
    elif category=='Chair':
        shapenet_ids = ['03001627','04256520']
    elif category=='Character':
        shapenet_ids=[]
    else:
        shapenet_ids=[]
        # raise ValueError('Category should be either Airplane, Character, or Chair (case-sensitive).')
    return Data(root,category,views,shape_list,shapenet_ids,batch_size=batch_size)
    return 

def load_encode_data(root,category,views,batch_size=-1):
    return 

if __name__=='__main__':

    # ./data/train/Character
    root = os.path.join(configs['TRAIN_DIR'],configs['CATEGORY'])
    
    train_data = load_train_data(root,configs['CATEGORY'],None)
    print()