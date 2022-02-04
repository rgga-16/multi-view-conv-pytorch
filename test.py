from tkinter import dnd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os,numpy as np

import data, network, seeder 
from args import args_

from seeder import init_fn
import utils as u
import ops

import datetime

def test(generator,discriminator,test_data):

    generator.eval()
    discriminator.eval()
    test_loader = DataLoader(test_data,batch_size=args_.batch_size,shuffle=True,worker_init_fn=init_fn)

    lambda_imloss = 1.0
    lambda_advloss = 0.01

    gen_running_loss = 0.0; gen_batch_running_loss=0.0
    disc_running_loss = 0.0; disc_batch_running_loss=0.0

    ct = datetime.datetime.now()
    out_dir = os.path.join(args_.out_dir,
                        args_.category,
                        'test',
                        f'[{ct.year}-{ct.month}-{ct.day} | {ct.hour}:{ct.minute}:{ct.second}]')

    for j,sample in enumerate(test_loader):
        
        input_sketches = sample[0]
        name = sample[1][0]
        with torch.set_grad_enabled(False):
            pred_raw = generator(input_sketches)

            bs,_,_,_ = pred_raw.shape

            pred_content = pred_raw[:,:-1,:,:] #Get content from prediction. Content is normal maps + depth maps
            pred_mask = pred_raw[:,-1:,:,:] #Get mask (-1,1) from prediction, which is the last channel.
            preds = u.apply_mask(pred_content,pred_mask) #Apply mask onto predicted content.

            preds_normal = pred_content[:,:-1,:,:] #Get predicted normal maps
            preds_depth = pred_content[:,-1:,:,:]

            for i in range(bs):
                pn = u.toti(preds_normal[i])
                pd = u.toti(preds_depth[i])
                pred = u.toti(preds[i])
                pm = u.toti(pred_mask[i])

                save_dir = os.path.join(out_dir,name)
                u.mkdir(save_dir)

                pn.save(os.path.join(save_dir,f'normal-dn{bs}--{i}.png'))
                pd.save(os.path.join(save_dir,f'depth-dn{bs}--{i}.png'))
                pred.save(os.path.join(save_dir,f'pred-dn{bs}--{i}.png'))
                pm.save(os.path.join(save_dir,f'mask-dn{bs}--{i}.png'))
            print(f'Images saved in {save_dir}.')
            
            
            # dn_list = torch.stack([itot(load_image(dn_f,'RGBA'),size=imsize) for dn_f in shape['dn_f']],dim=0)
            # dnfs_list = torch.stack([itot(load_image(dnfs_f,'RGBA'),size=imsize) for dnfs_f in shape['dnfs_f']],dim=0)
            # targets_list = torch.cat([dnfs_list,dn_list],dim=0)
            # real_mask = bool_to_real_mask(bool_masks)
            # targets_list = torch.cat([targets_list,real_mask],dim=1)

    return 



if __name__=='__main__':

    gen_wts_f = os.path.join(args_.weights_dir,'best_gen_weights.pth')
    disc_wts_f = os.path.join(args_.weights_dir,'best_disc_weights.pth')
    
    gen_wts = torch.load(gen_wts_f,map_location=u.device)
    disc_wts = torch.load(disc_wts_f,map_location=u.device)

    sketch_in_c = len(args_.sketch_views)*2

    generator = network.MonsterNet(n_target_views=args_.num_dn_views+args_.num_dnfs_views,in_c=sketch_in_c)
    # d_in_c = 4+sketch_in_c
    d_in_c = sketch_in_c
    discriminator = network.Discriminator(in_c=d_in_c)
    
    root = os.path.join(args_.test_dir,args_.category)

    test_data = data.load_test_data(root,args_.category,None)

    test(generator,discriminator,test_data)