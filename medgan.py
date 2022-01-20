"Trainer for MedGAN"
import torch.nn as nn
import torch
from network import Dis, Dis_aux, G, Encoder, Decoder, Seg,init_weights,ImagePool,get_scheduler
from loss import DiceCeLoss
from torchvision.utils import save_image
import os

torch.autograd.set_detect_anomaly(True)

def denorm(img):
    #convert [-1,1] to [0,1]
    #to use torchvision.utils.save_image
    return (img+1.0)/2

class MedGAN(nn.Module):
    def __init__(self,params):
        super().__init__()
        num_classes = params['train']['num_classes']
        lr_seg = params['train']['lr_seg']
        lr = params['train']['lr']
        self.skip = params['train']['skip']
        #network
        self.gen = G(skip=self.skip)
        self.enc = Encoder()
        self.dec = Decoder(skip=self.skip)
        self.seg = Seg(num_classes)
        self.disA = Dis_aux()
        self.disB = Dis()
        self.disSeg = Dis(num_classes)
        #optimizer
        self.gen_opt = torch.optim.Adam(self.gen.parameters(),lr=lr,betas=(0.5,0.999))
        self.enc_opt = torch.optim.Adam(self.enc.parameters(),lr=lr,betas=(0.5,0.999),weight_decay=0.0001)
        self.dec_opt = torch.optim.Adam(self.dec.parameters(),lr=lr,betas=(0.5,0.999))
        self.seg_opt = torch.optim.Adam(self.seg.parameters(),lr=lr_seg,weight_decay=0.0001)
        self.disA_opt = torch.optim.Adam(self.disA.parameters(),lr=lr,betas=(0.5,0.999))
        self.disB_opt = torch.optim.Adam(self.disB.parameters(),lr=lr,betas=(0.5,0.999))
        self.disSeg_opt = torch.optim.Adam(self.disSeg.parameters(),lr=lr,betas=(0.5,0.999))
        # fake image pool
        self.fakeA_pool = ImagePool()
        self.fakeB_pool = ImagePool()
        # lr update
        self.seg_opt_sch = get_scheduler(self.seg_opt)
        #loss
        self.segloss = DiceCeLoss(num_classes)
        self.criterionL2 = nn.MSELoss()
        self.criterionL1 = nn.L1Loss()