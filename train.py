
import torch
from torch.utils.data import DataLoader
import os

import data, network, utils,seeder 

from seeder import init_fn
from utils import configs


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
                sketch = sample[0]; dn = sample[1]; dnfs = sample[2]
                print()
                with torch.set_grad_enabled(phase=='train'):
                    output = model(sketch)
                    print()
                    # Calculate losses
                    # Loss.backward()
                    # Optim.step()


    return

if __name__=='__main__':

    root = os.path.join(configs['TRAIN_DIR'],configs['CATEGORY'])
    
    train_data = data.load_train_data(root,configs['CATEGORY'],None)

    n_sketch_views = len(configs['SKETCH_VIEWS']) * configs['SKETCH_VARIATIONS']

    model = network.MonsterNet(n_sketch_views=n_sketch_views,n_dn_views=configs['NUM_DN_VIEWS'])

    train_loop(model,train_data,None)


    print()