# Computation of losses and other operations here
import torch

def overall_loss():

    return 

'''
Per-pixel depth_loss 
'''
def depth_loss(pred,real):
    l1_dist = torch.abs(pred-real)
    return 

'''
Per-pixel normal_loss 
'''
def normal_loss(pred,real):
    l1_dist = torch.abs(pred-real)
    return 

def mask_loss(pred,real):

    return 

def adversarial_loss(pred,real):

    return 