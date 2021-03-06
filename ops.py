# Computation of losses and other operations here
import torch
import torch.nn.functional as F
from utils import device,denormalize

def overall_loss():

    return 

'''
Per-pixel depth_loss 
'''
def depth_loss(pred,target,bool_mask,normalized=True):
    """
    input:
        predicts   : n x H x W x 1      predicted depths
        targets    : n x H x W x 1      ground-truth depths
        mask       : n x H x W x 1     bool mask, included to only account for losses on foreground
        normalized : boolean        whether output loss should be normalized by pixel number
    output:
        loss       : scalar             loss value
	"""

    b,c,_,_ = pred.shape
    l1_dist = torch.abs(pred-target)
    l1_dist_masked = torch.masked_select(l1_dist,bool_mask)

    if normalized:
        depth_loss = torch.mean(l1_dist_masked) * (b*c)
        
    else: 
        depth_loss = torch.sum(l1_dist_masked)

    return depth_loss

'''
Per-pixel normal_loss 
'''
def normal_loss(pred,target,bool_mask,normalized=True):
    """
        input:
            predicts   : n x H x W x 3      predicted normals
            targets    : n x H x W x 3      ground-truth normals
            mask       : n x H x W x 1      boolean mask. Applied to only account for loss values from foreground
            normalized : boolean            whether output loss should be normalized by pixel number
        output:
            loss       : scalar             loss value
    """

    b,c,_,_ = pred.shape 
    l2_dist = torch.square(pred-target)
    l2_dist_masked = torch.masked_select(l2_dist,bool_mask)

    if normalized:
        normal_loss = torch.mean(l2_dist_masked) * (b*c)
    else: 
        normal_loss = torch.sum(l2_dist_masked)

    return normal_loss

'''
Binary cross entropy loss between predicted and target binary mask
'''
def mask_loss(pred,target,normalized=True):

    # mask_loss = F.binary_cross_entropy(pred,target)
    p = denormalize(pred)
    t = denormalize(target)
    mask_loss = F.binary_cross_entropy(p,t,reduction='mean')

    return mask_loss

def adversarial_loss(pred,target,discriminator):

    target_probs = discriminator(target) #Get probablities on target/real batch
    real_label_d = torch.full_like(target_probs,1.0,device=device).detach()
    loss_d_real = F.binary_cross_entropy(target_probs,real_label_d) #Compute discriminator error on target batch that it is real
    
    pred_probs = discriminator(pred) #Get probabilities on predicted/fake batch
    fake_label_d = torch.full_like(pred_probs,0.0,device=device).detach()
    loss_d_fake = F.binary_cross_entropy(pred_probs.detach(),fake_label_d) #Compute discriminator error on predicted batch that it is fake

    loss_d = loss_d_real + loss_d_fake

    real_label_g = torch.full_like(pred_probs,1.0,device=device).detach()
    pred_probs_real = discriminator(pred)
    loss_g = F.binary_cross_entropy(pred_probs_real,real_label_g) #Compute error on predicted batch that it is real. For the generator
    
    return loss_g,loss_d

def adversarial_loss2(disc_data,discriminator):

    probs = discriminator(disc_data)
    split_size = probs.shape[0]//2
    probs_split = torch.split(probs,split_size,dim=0)
    probs_targets = probs_split[0]; probs_preds = probs_split[1]

    loss_g = torch.sum(-1.0 * torch.log(probs_preds))
    loss_d_r = torch.sum(-1.0 * torch.log(probs_targets))
    loss_d_f = torch.sum(-1.0 * torch.log(1-probs_preds))

    return loss_g, loss_d_r, loss_d_f