import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os,copy,numpy as np

import data, network, utils,seeder 
import args
from args import args_

from seeder import init_fn
import utils as u

import datetime

def infer(gen_path,chkpt_path,im_paths):

    sketch_in_c = len(args_.sketch_views)*2
    generator = network.MonsterNet(args_.num_dn_views+args_.num_dnfs_views,in_c=sketch_in_c)

    imsize = args_.imsize

    sketch_list_init = torch.stack([u.itot(u.load_image(im_f,'L'),size=imsize) for im_f in im_paths],dim=1).squeeze(0)
    sketch_list_flipped = torch.flip(sketch_list_init,[1,2])
    input_sketches = torch.cat([sketch_list_init,sketch_list_flipped],dim=0)

    ct = datetime.datetime.now()
    out_dir = os.path.join(args_.out_dir,
                        args_.category,
                        'infer',
                        f'[{ct.year}-{ct.month}-{ct.day} | {ct.hour}:{ct.minute}:{ct.second}]')
    u.mkdir(out_dir)

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

            pn.save(os.path.join(out_dir,f'normal-dn{bs}--{i}.png'))
            pd.save(os.path.join(out_dir,f'depth-dn{bs}--{i}.png'))
            pred.save(os.path.join(out_dir,f'pred-dn{bs}--{i}.png'))
            pm.save(os.path.join(out_dir,f'mask-dn{bs}--{i}.png'))
        print(f'Images saved in {out_dir}.')


    



    return 