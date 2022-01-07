# Computation of losses and other operations here
import torch
import torch.nn.functional as F
from utils import device

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
        print()
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

def mask_loss(pred,target,normalized=True):
    bce = torch.nn.BCELoss()
    # mask_loss = F.binary_cross_entropy(pred,target)
    mask_loss = bce(pred,target).item() #CUDA ERROR HERE. Try to debug this.

    p = pred*0.5 + 0.5
    t = target*0.5 + 0.5

    # mask_loss2 = -t * log(p) - (1-t)*(log(1-p))
    mask_loss2 = torch.sum(-1.0 * torch.multiply(torch.log(torch.max(torch.tensor(1e-6),p)),t) - torch.multiply(1-t, torch.log(torch.max(torch.tensor(1e-6),1-p))))

    if normalized:
        b,c,h,w = pred.shape
        n_pixels = c*h*w 
        mask_loss/=n_pixels
    return mask_loss

def adversarial_loss(pred,target):

    return 