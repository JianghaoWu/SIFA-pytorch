"Trainer for SIFA"
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



class SIFA(nn.Module):
    def __init__(self,params):
        super().__init__()
        num_classes = params['train']['num_classes']
        lr_seg = params['train']['lr_seg']
        lr = params['train']['lr']
        # self.skip = True
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

    def initialize(self):
        init_weights(self.gen)
        init_weights(self.dec)
        init_weights(self.enc)
        init_weights(self.seg)
        init_weights(self.disA)
        init_weights(self.disB)
        init_weights(self.disSeg)

    def forward(self):
        self.fakeB = self.gen(self.realA, self.realA)
        self.latent_realB = self.enc(self.realB)
        self.latent_fakeB = self.enc(self.fakeB)
        self.fakeA = self.dec(self.latent_realB, self.realB)
        self.pred_mask_b = self.seg(self.latent_realB)
        self.pred_mask_fake_b = self.seg(self.latent_fakeB)
        self.cycleA = self.dec(self.latent_fakeB, self.fakeB)
        self.cycleB = self.gen(self.fakeA, self.fakeA)

    def backward_D(self, netD, real, fake,aux=False):
        if aux:
            pred_real = netD.forward_aux(real)
            pred_fake = netD.forward_aux(fake.detach())
        else:
            pred_real = netD(real)
            pred_fake = netD(fake.detach())        
        all1 = torch.ones_like(pred_real)
        all0 = torch.zeros_like(pred_fake)
        loss_real = self.criterionL2(pred_real, all1)
        loss_fake = self.criterionL2(pred_fake, all0)
        loss_D = (loss_real + loss_fake) * 0.5
        return loss_D


    def backward_G(self, netD, fake,aux=False):
        if aux:
            out = netD.forward_aux(fake)
        else:
            out = netD(fake)
        all1 = torch.ones_like(out)
        loss_G = self.criterionL2(out, all1)
        return loss_G
        
    def update_lr(self):
        self.seg_opt_sch.step()

    def update_GAN(self,imagesa,imagesb):
        self.realA = imagesa
        self.realB = imagesb
        self.forward()
        #update DisA
        self.disA_opt.zero_grad()
        self.fakeA_from_pool = self.fakeA_pool.query(self.fakeA)
        loss_disA = self.backward_D(self.disA,self.realA,self.fakeA_from_pool)
        loss_disA_aux = self.backward_D(self.disA,self.cycleA.detach(),self.fakeA_from_pool,aux=True)
        loss_disA = loss_disA + loss_disA_aux
        loss_disA.backward()
        self.disA_opt.step()
        #update disB
        self.disB_opt.zero_grad()
        self.fakeB_from_pool = self.fakeB_pool.query(self.fakeB)
        loss_disB = self.backward_D(self.disB,self.realB,self.fakeB_from_pool)
        loss_disB.backward()
        self.disB_opt.step()
        #update DisSeg
        self.disSeg_opt.zero_grad()
        loss_disSeg = self.backward_D(self.disSeg,self.pred_mask_fake_b.detach(),self.pred_mask_b)
        loss_disSeg.backward()
        self.disSeg_opt.step()
        #update G
        self.gen_opt.zero_grad()
        self.dec_opt.zero_grad()
        loss_cycleA = self.criterionL1(self.realA,self.cycleA) * 10.0
        loss_cycleB = self.criterionL1(self.realB,self.cycleB) * 10.0
        g_loss = loss_cycleA + loss_cycleB + self.backward_G(self.disB,self.fakeB) + self.backward_G(self.disA,self.fakeA)
        g_loss.backward()
        self.gen_opt.step()
        self.dec_opt.step()
        
        self.loss_cyclea = loss_cycleA.item()
        self.loss_cycleb = loss_cycleB.item()
        
    def update_seg(self,imagesa,imagesb,labelsa):
        self.realA = imagesa
        self.realB = imagesb
        self.labelA = labelsa
        self.forward()
        #update encoder and seg
        self.enc_opt.zero_grad()
        self.seg_opt.zero_grad()
        seg_loss_B = self.segloss(self.pred_mask_fake_b,self.labelA) + \
             self.criterionL1(self.realA,self.cycleA) + self.criterionL1(self.realB, self.cycleB) + \
             0.1 * self.backward_G(self.disA,self.fakeA) + 0.1 * self.backward_G(self.disSeg,self.pred_mask_b) + 0.1 * self.backward_G(self.disA,self.fakeA,aux=True)
        
        seg_loss_B.backward()
        self.enc_opt.step()
        self.seg_opt.step()
        self.loss_seg = self.segloss(self.pred_mask_fake_b,self.labelA).item()

    def test_seg(self,imagesb):
        #for test
        content_b = self.enc(imagesb)
        pred_mask_b = self.seg(content_b)
        return pred_mask_b

    def sample_image(self, epoch, exp_name):
        sample_image_dir = "sample_image/" + str(exp_name)
        if(not os.path.exists(sample_image_dir)):
            os.mkdir(sample_image_dir)
        print(sample_image_dir,epoch+1)
        save_image(denorm(self.realA), '{}/realA-epoch-{}.jpg'.format(sample_image_dir, epoch + 1))
        save_image(denorm(self.realB), '{}/realB-epoch-{}.jpg'.format(sample_image_dir, epoch + 1))
        save_image(denorm(self.fakeA), '{}/fakeA-epoch-{}.jpg'.format(sample_image_dir, epoch + 1))
        save_image(denorm(self.fakeB), '{}/fakeB-epoch-{}.jpg'.format(sample_image_dir, epoch + 1))
        save_image(denorm(self.cycleA), '{}/cycleA-epoch-{}.jpg'.format(sample_image_dir, epoch + 1))
        save_image(denorm(self.cycleB), '{}/cycleB-epoch-{}.jpg'.format(sample_image_dir, epoch + 1))

    def print_loss(self):
        print('---------------------')
        print('loss cycle A:', self.loss_cyclea)
        print('loss cycle B:', self.loss_cycleb)
        print('loss seg:', self.loss_seg)
        return self.loss_cyclea,self.loss_cycleb,self.loss_seg


