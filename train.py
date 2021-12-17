
import torch 


def train_step(model,dataloader,loss_history,optim=None):


    return 


def train_loop(model, train_data, val_data, n_epochs=10, batch_size=12, lr=0.01): 

    train_loader = None
    val_loader = None

    optimizer = None

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
                with torch.set_grad_enabled(phase=='train'):
                    output = model.forward()
                    # Calculate losses
                    # Loss.backward()
                    # Optim.step()




    

    return