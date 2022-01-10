
import torch
from torch.utils.data import DataLoader
import os

import data, network, utils,seeder 

from seeder import init_fn
from utils import configs,device,apply_mask,show_images,real_to_bool_mask
from ops import depth_loss,normal_loss,adversarial_loss,mask_loss,overall_loss,adversarial_loss


def train_step(model,dataloader,loss_history,optim=None):


    return 


def train_loop(model, train_data, val_data,discriminator=None): 
    n_epochs = configs['EPOCHS']
    train_loader = DataLoader(train_data,batch_size = configs['BATCH_SIZE'], shuffle=True,worker_init_fn=init_fn)
    val_loader = DataLoader(train_data,batch_size = configs['BATCH_SIZE'], shuffle=True,worker_init_fn=init_fn)

    gen_optimizer = torch.optim.Adam(model.parameters(),lr=configs['LR'],betas=(0.9,0.999))
    if discriminator:
        disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=configs['LR'],betas=(0.9,0.999))
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_optimizer,gamma=0.96,verbose=True)
    
    gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,gamma=0.96,verbose=True)

    train_loss_history = []
    val_loss_history = []

    lambda_imloss = 1.0
    lambda_advloss = 0.01
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
                    # tm_min = torch.min(target_mask)
                    # tm_max = torch.max(target_mask)
                    targets = apply_mask(target_content,target_mask)

                    isketches_expanded = torch.tile(input_sketches,(target_raw.shape[0],1,1,1))

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

                    if discriminator:
                        disc_data = torch.cat([targets,preds],dim=0)
                        se = torch.cat([isketches_expanded,isketches_expanded],dim=0)
                        disc_data = torch.cat([se,disc_data],dim=1)
                        loss_g_a,loss_d_r,loss_d_f = adversarial_loss(disc_data,discriminator)
                    else: 
                        loss_g_a=0
                        loss_d_r=0
                        loss_d_f=0


                    overall_g_loss = (lambda_imloss * (d_loss+n_loss+m_loss)) + lambda_advloss * loss_g_a
                    overall_d_loss = loss_d_r + loss_d_f

                    overall_g_loss.backward()
                    overall_d_loss.backward()

                    # Implement checkpoint and printing losses and then train!


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
    sketch_in_c = len(configs['SKETCH_VIEWS'])*2

    model = network.MonsterNet(n_target_views=configs['NUM_DN_VIEWS']+configs['NUM_DNFS_VIEWS'],in_c=sketch_in_c)
    d_in_c = 4+sketch_in_c
    discriminator = network.Discriminator(in_c=4+sketch_in_c)
    train_loop(model,train_data,None,discriminator)


    print()