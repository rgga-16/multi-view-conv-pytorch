
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os,copy,numpy as np

import data, network, utils,seeder 

from seeder import init_fn
from utils import configs,device,apply_mask,show_images,real_to_bool_mask
from ops import adversarial_loss2, depth_loss,normal_loss,adversarial_loss,mask_loss,overall_loss,adversarial_loss


def train_step(model,dataloader,loss_history,optim=None):


    return 


def train_loop(generator, train_data, val_data,discriminator=None,gen_chkpt_path=None,disc_chkpt_path=None): 
    
    n_epochs = configs['EPOCHS']
    start_epoch=0
    
    best_gen_wts = copy.deepcopy(generator.state_dict())
    best_val_gen_l = np.inf

    if discriminator:
        best_disc_wts = copy.deepcopy(discriminator.state_dict())
        best_val_disc_l = np.inf

    train_gen_lh = []
    train_disc_lh = []
    val_gen_lh = []
    val_disc_lh = []
    epoch_chkpts = []
    gen_epoch_lh = []
    disc_epoch_lh = []

    train_loader = DataLoader(train_data,batch_size = configs['BATCH_SIZE'], shuffle=True,worker_init_fn=init_fn)
    val_loader = DataLoader(val_data,batch_size = configs['BATCH_SIZE'], shuffle=True,worker_init_fn=init_fn)

    gen_optimizer = torch.optim.Adam(generator.parameters(),lr=configs['LR'],betas=(0.9,0.999))
    if discriminator:
        disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=configs['LR'],betas=(0.9,0.999))
        disc_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_optimizer,gamma=0.96,verbose=True)
    
    gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_optimizer,gamma=0.96,verbose=True)

    lambda_imloss = 1.0
    lambda_advloss = 0.01

    if gen_chkpt_path:
        gen_chkpt = torch.load(gen_chkpt_path)
        generator.load_state_dict(gen_chkpt)
        start_epoch = gen_chkpt['epoch']+1
        epoch_chkpts = gen_chkpt['epoch_chkpts']

        best_gen_wts = gen_chkpt['best_gen_weights']
        best_val_gen_l = gen_chkpt['best_val_gen_l']
        train_gen_lh = gen_chkpt['train_gen_lh']
        val_gen_lh = gen_chkpt['val_gen_lh']
        # gen_epoch_lh = gen_chkpt['gen_epoch_lh']
        print(f'Generator checkpoint found. Resuming training from epoch {start_epoch+1}.\n')
    
    if discriminator and disc_chkpt_path:
        disc_chkpt = torch.load(disc_chkpt_path)
        discriminator.load_state_dict(disc_chkpt)

        best_disc_wts = disc_chkpt['best_disc_weights']
        best_val_disc_l = disc_chkpt['best_val_disc_l']
        train_disc_lh = disc_chkpt['train_disc_lh']
        val_disc_lh = disc_chkpt['val_disc_lh']
        # disc_epoch_lh = disc_chkpt['disc_epoch_lh']
        print(f'Discriminator checkpoint found.')


    for epoch in range(start_epoch,n_epochs):
        for phase in ['train','val']:
            if phase=='train':
                generator.train()
                dataloader = train_loader 
                gen_lh = train_gen_lh
                if discriminator: disc_lh = train_disc_lh
            else:
                generator.eval()
                dataloader = val_loader
                gen_lh = val_gen_lh
                if discriminator: disc_lh = val_disc_lh
            
            #Print losses every b batches
            b = 1 if len(dataloader) <= configs['NUM_BATCH_CHKPTS'] else len(dataloader)//configs['NUM_BATCH_CHKPTS']

            gen_running_loss, gen_batch_running_loss = 0.0,0.0
            disc_running_loss, disc_batch_running_loss = 0.0,0.0
            for i,sample in enumerate(dataloader):
                with torch.autograd.set_detect_anomaly(True):
                    input_sketches = sample[0]
                    target_raw = sample[1]

                    tb = target_raw.shape[0]
                    target_raw = torch.flatten(target_raw,start_dim=0,end_dim=1)

                    gen_optimizer.zero_grad()
                    if discriminator: disc_optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        pred_raw = generator(input_sketches)
                        pred_content = pred_raw[:,:-1,:,:] #Get content from prediction. Content is normal maps + depth maps
                        pred_mask = pred_raw[:,-1:,:,:] #Get mask (-1,1) from prediction, which is the last channel.
                        preds = apply_mask(pred_content,pred_mask) #Apply mask onto predicted content.

                        target_content = target_raw[:,:-1,:,:]             
                        target_mask = target_raw[:,-1:,:,:]
                        targets = apply_mask(target_content,target_mask)

                        if configs['PREDICT_NORMAL']:
                            preds_normal = pred_content[:,:-1,:,:] #Get predicted normal maps
                            preds_depth = pred_content[:,-1:,:,:]
                            targets_normal = target_content[:,:-1,:,:]
                            targets_depth = target_content[:,-1:,:,:]
                        else: 
                            preds_depth = pred_content
                            preds_normal = torch.tile(torch.zeros_like(preds_depth,device=device),(1,3,1,1))
                            targets_depth = target_content
                            targets_normal = torch.tile(torch.zeros_like(targets_depth,device=device),(1,3,1,1))

                        L_depth = depth_loss(preds_depth,targets_depth,real_to_bool_mask(target_mask))
                        L_normal = normal_loss(preds_normal,targets_normal,real_to_bool_mask(target_mask))
                        L_mask = mask_loss(pred_mask,target_mask)

                        if discriminator:
                            L_adv_g,L_adv_d = adversarial_loss(preds,targets,discriminator)
                        else: 
                            L_adv_g=0
                            L_adv_d = 0
                        
                        L_G = (lambda_imloss * (L_depth+L_normal+L_mask)) + lambda_advloss * L_adv_g


                        if phase=='train':
                            L_G.backward()
                            gen_optimizer.step()
                            del pred_mask; del target_mask
                            del preds_normal; del targets_normal
                            del preds_depth; del targets_depth
                            del pred_content; del target_content

                            if discriminator:
                                L_adv_d.backward()
                                disc_optimizer.step()

                        gen_running_loss+= L_G.item() *tb
                        disc_running_loss += L_adv_d.item() * tb
                        gen_batch_running_loss += L_G.item()
                        disc_batch_running_loss += L_adv_d.item()
                        
                        gen_lh.append(L_G.item())
                        if discriminator: disc_lh.append(L_adv_d.item())

                        if i % b == b-1: 
                            batch_str = f'[{phase} Batch {i+1}/{len(dataloader)}] Gen Loss: {gen_batch_running_loss/b:.5f}'
                            if discriminator: batch_str = f'{batch_str} | Disc Loss: {disc_batch_running_loss/b:.5f}'
                            gen_batch_running_loss = 0.0; disc_batch_running_loss=0.0
                            print(batch_str)
            
            gen_epoch_loss = gen_running_loss/ dataloader.dataset.__len__()
            disc_epoch_loss = disc_running_loss / dataloader.dataset.__len__() 
            print(f'{phase} Gen Loss: {gen_epoch_loss:.5f}')
            if discriminator: print(f'{phase} Disc Loss: {disc_epoch_loss:.5f}')

            if phase =='val' and gen_epoch_loss < best_val_gen_l:
                best_val_gen_l = gen_epoch_loss
                best_gen_wts = copy.deepcopy(generator.state_dict())
                print(f'Found best generator params at epoch {epoch+1}')
            
            if discriminator:
                if phase =='val' and disc_epoch_loss < best_val_disc_l:
                    best_val_disc_l = disc_epoch_loss 
                    best_disc_wts = copy.deepcopy(discriminator.state_dict())
                    print(f'Found best discriminator params at epoch {epoch+1}')
        
        epoch_chkpts.append(epoch)
            # gen_epoch_lh.append(gen_epoch_loss)
            # disc_epoch_lh.append(disc_epoch_loss)

        gen_chkpt = generator.state_dict()
        gen_chkpt['epoch'] = epoch
        gen_chkpt['best_gen_weights'] = best_gen_wts
        gen_chkpt['best_val_gen_l'] = best_val_gen_l
        gen_chkpt['train_gen_lh'] = train_gen_lh
        gen_chkpt['val_gen_lh'] = val_gen_lh
        gen_chkpt['epoch_chkpts'] = epoch_chkpts
        gen_chkpt_path = os.path.join(configs['CHKPTS_DIR'],'gen_chkpt.pt')
        torch.save(gen_chkpt,gen_chkpt_path)

        if discriminator: 
            disc_chkpt = discriminator.state_dict()
            disc_chkpt['best_disc_weights'] = best_disc_wts
            disc_chkpt['best_val_disc_l'] = best_val_disc_l
            disc_chkpt['train_disc_lh'] = train_disc_lh
            disc_chkpt['val_disc_lh'] = val_disc_lh
            disc_chkpt_path = os.path.join(configs['CHKPTS_DIR'],'disc_chkpt.pt')
            torch.save(disc_chkpt,disc_chkpt_path)
            print(f'Discriminator checkpoint saved in {disc_chkpt_path}')
        
        print(f'Generator checkpoint saved in {gen_chkpt_path}')
        
        if epoch % 10 ==10-1:
            gen_lr_scheduler.step()
            if discriminator: disc_lr_scheduler.step()


    return

if __name__=='__main__':

    root = os.path.join(configs['TRAIN_DIR'],configs['CATEGORY'])
    
    train_data = data.load_train_data(root,configs['CATEGORY'],None)
    val_data = data.load_train_data(root,configs['CATEGORY'],None)

    # n_sketch_views = len(configs['SKETCH_VIEWS']) * configs['SKETCH_VARIATIONS']
    sketch_in_c = len(configs['SKETCH_VIEWS'])*2

    generator = network.MonsterNet(n_target_views=configs['NUM_DN_VIEWS']+configs['NUM_DNFS_VIEWS'],in_c=sketch_in_c)
    # d_in_c = 4+sketch_in_c
    d_in_c = sketch_in_c
    discriminator = network.Discriminator(in_c=d_in_c)
    train_loop(generator,train_data,val_data,discriminator)


    