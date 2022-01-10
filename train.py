
import torch
from torch.utils.data import DataLoader
import os

import data, network, utils,seeder 

from seeder import init_fn
from utils import configs,device,apply_mask,show_images,real_to_bool_mask
from ops import depth_loss,normal_loss,adversarial_loss,mask_loss,overall_loss


def train_step(model,dataloader,loss_history,optim=None):


    return 


def train_loop(model, train_data, val_data): 
    n_epochs = configs['EPOCHS']
    train_loader = DataLoader(train_data,batch_size = configs['BATCH_SIZE'], shuffle=True,worker_init_fn=init_fn)
    val_loader = DataLoader(train_data,batch_size = configs['BATCH_SIZE'], shuffle=True,worker_init_fn=init_fn)

    optimizer = torch.optim.Adam(model.parameters(),lr=configs['LR'],betas=(0.9,0.999))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.96,verbose=True)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(n_epochs):
        for phase in ['train','val']:
            if phase=='train':
                model.train()
                dataloader = train_loader 
            else:
                model.eval()
                dataloader = val_loader

            for i,sample in enumerate(dataloader):
                input_sketches = sample[0]; target_raw = sample[1].view((sample[1].shape[0]*sample[1].shape[1],sample[1].shape[2],sample[1].shape[3],-1))

                with torch.set_grad_enabled(phase=='train'):
                    pred_raw = model(input_sketches)
                    pred_content = pred_raw[:,:-1,:,:]
                    pred_mask = pred_raw[:,-1:,:,:]
                    preds = apply_mask(pred_content,pred_mask)

                    target_content = target_raw[:,:-1,:,:]             
                    target_mask = target_raw[:,-1:,:,:]
                    mask_max = torch.max(target_mask)
                    mask_min = torch.min(target_mask)
                    targets = apply_mask(target_content,target_mask)

                    if configs['PREDICT_NORMAL']:
                        preds_normal = pred_content[:,:-1,:,:]
                        preds_depth = pred_content[:,-1:,:,:]
                        targets_normal = target_content[:,:-1,:,:]
                        targets_depth = target_content[:,-1:,:,:]
                        pass 
                    else: 
                        preds_depth = pred_content
                        preds_normal = torch.tile(torch.zeros_like(preds_depth,device=device),(1,3,1,1))
                        targets_depth = target_content
                        targets_normal = torch.tile(torch.zeros_like(targets_depth,device=device),(1,3,1,1))

                    d_loss = depth_loss(preds_depth,targets_depth,real_to_bool_mask(target_mask))
                    n_loss = normal_loss(preds_normal,targets_normal,real_to_bool_mask(target_mask))
                    m_loss = mask_loss(pred_mask,target_mask)

                    # show_images(target_content)
                    # show_images(target_mask)
                    # show_images(targets)
                    print()



                    # Predicted output is the first n_channel layers. last channel is mask. 
                    # preds_content = tf.slice(preds, [0,0,0,0], [-1,-1,-1,num_channels-1])
		            # preds_mask = tf.slice(preds, [0,0,0,num_channels-1], [-1,-1,-1,1])
		            # preds = image.apply_mask(preds_content, preds_mask)
                    print()
                    # Calculate losses
                    # Loss.backward()
                    # Optim.step()


    return

if __name__=='__main__':

    root = os.path.join(configs['TRAIN_DIR'],configs['CATEGORY'])
    
    train_data = data.load_train_data(root,configs['CATEGORY'],None)

    # n_sketch_views = len(configs['SKETCH_VIEWS']) * configs['SKETCH_VARIATIONS']
    in_c = len(configs['SKETCH_VIEWS'])*2

    model = network.MonsterNet(n_target_views=configs['NUM_DN_VIEWS']+configs['NUM_DNFS_VIEWS'],in_c=in_c)

    train_loop(model,train_data,None)


    print()