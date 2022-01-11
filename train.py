
import torch
from torch.utils.data import DataLoader
import os,copy,numpy as np

import data, network, utils,seeder 

from seeder import init_fn
from utils import configs,device,apply_mask,show_images,real_to_bool_mask
from ops import depth_loss,normal_loss,adversarial_loss,mask_loss,overall_loss,adversarial_loss


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

    if gen_chkpt_path and disc_chkpt_path:
        gen_chkpt = torch.load(gen_chkpt_path)
        generator.load_state_dict(gen_chkpt)
        start_epoch = gen_chkpt['epoch']+1
        epoch_chkpts = gen_chkpt['epoch_chkpts']

        best_gen_wts = gen_chkpt['best_gen_weights']
        best_val_gen_l = gen_chkpt['best_val_gen_l']
        train_gen_lh = gen_chkpt['train_gen_lh']
        val_gen_lh = gen_chkpt['val_gen_lh']
        # gen_epoch_lh = gen_chkpt['gen_epoch_lh']

        disc_chkpt = torch.load(disc_chkpt_path)
        discriminator.load_state_dict(disc_chkpt)

        best_disc_wts = disc_chkpt['best_disc_weights']
        best_val_disc_l = disc_chkpt['best_val_disc_l']
        train_disc_lh = disc_chkpt['train_disc_lh']
        val_disc_lh = disc_chkpt['val_disc_lh']
        # disc_epoch_lh = disc_chkpt['disc_epoch_lh']

        print(f'Generator & Discriminator checkpoints found. Resuming training from epoch {start_epoch+1}.\n')


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
                input_sketches = sample[0]
                # target_raw = sample[1].view((sample[1].shape[0]*sample[1].shape[1],sample[1].shape[2],sample[1].shape[3],-1))
                target_raw = sample[1]
                tb,tn_views,tc,th,tw = target_raw.shape
                target_raw = target_raw.view(tb*tn_views,tc,th,tw)
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

                    isketches_expanded = torch.tile(input_sketches,(target_raw.shape[0],1,1,1))

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

                    if phase=='train':
                        overall_g_loss.backward()
                        overall_d_loss.backward() #BUG HERE. Cannot backward overall_d_loss(). I think you shouldn't input the disc_data concatenated together.
                        # Instead, input the real batch and fake batch separately into the discriminator.
                        gen_optimizer.step()
                        if discriminator: disc_optimizer.step()
                    
                    gen_running_loss+= overall_g_loss *tb
                    disc_running_loss += overall_d_loss * tb
                    gen_batch_running_loss += overall_g_loss
                    disc_batch_running_loss += overall_d_loss
                    
                    gen_lh.append(overall_g_loss)
                    disc_lh.append(overall_d_loss)

                    # show_images(target_content)
                    # show_images(target_mask)
                    # show_images(targets)
                    if i % b == b-1: 
                        batch_str = f'[{phase} Batch {i+1}/{len(dataloader)}] Gen Loss: {gen_batch_running_loss/b:.5f} | Disc Loss: {disc_batch_running_loss/b:.5f}'
                        gen_batch_running_loss = 0.0; disc_batch_running_loss=0.0
                        print(batch_str)

            
            gen_epoch_loss = gen_running_loss/ dataloader.dataset.__len__()
            disc_epoch_loss = disc_running_loss / dataloader.dataset.__len__() 
            print(f'{phase} Gen Loss: {gen_epoch_loss:.5f}')
            print(f'{phase} Disc Loss: {disc_epoch_loss:.5f}')

            if phase =='val' and gen_epoch_loss < best_val_gen_l:
                best_val_gen_l = gen_epoch_loss
                best_gen_wts = copy.deepcopy(generator.state_dict())
                print(f'Found best generator params at epoch {epoch+1}')
            
            if phase =='val' and disc_epoch_loss < best_val_disc_l:
                best_val_disc_l = disc_epoch_loss 
                best_disc_wts = copy.deepcopy(discriminator.state_dict())
                print(f'Found best discriminator params at epoch {epoch+1}')
        
        epoch_chkpts.append(epoch)
            # gen_epoch_lh.append(gen_epoch_loss)
            # disc_epoch_lh.append(disc_epoch_loss)

        gen_chkpt = generator.get_state_dict()
        gen_chkpt['epoch'] = epoch
        gen_chkpt['best_gen_weights'] = best_gen_wts
        gen_chkpt['best_val_gen_l'] = best_val_gen_l
        gen_chkpt['train_gen_lh'] = train_gen_lh
        gen_chkpt['val_gen_lh'] = val_gen_lh
        gen_chkpt['epoch_chkpts'] = epoch_chkpts

        disc_chkpt = discriminator.get_state_dict()
        disc_chkpt['best_disc_weights'] = best_disc_wts
        disc_chkpt['best_val_disc_l'] = best_val_disc_l
        disc_chkpt['train_disc_lh'] = train_disc_lh
        disc_chkpt['val_disc_lh'] = val_disc_lh

        gen_chkpt_path = os.path.join(configs['CHKPTS_DIR'],'gen_chkpt.pt')
        disc_chkpt_path = os.path.join(configs['CHKPTS_DIR'],'disc_chkpt.pt')

        torch.save(gen_chkpt,gen_chkpt_path)
        torch.save(disc_chkpt,disc_chkpt_path)
        print(f'Generator checkpoint saved in {gen_chkpt_path}')
        print(f'Discriminator checkpoint saved in {disc_chkpt_path}')
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
    d_in_c = 4+sketch_in_c
    discriminator = network.Discriminator(in_c=4+sketch_in_c)
    train_loop(generator,train_data,val_data,discriminator)


    print()