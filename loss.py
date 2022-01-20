#loss function for SIFA
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
import numpy as np


    
def dice_loss(predict,target):
    target = target.float()
    smooth = 1e-4
    intersect = torch.sum(predict*target)
    dice = (2 * intersect + smooth)/(torch.sum(target)+torch.sum(predict*predict)+smooth)
    loss = 1.0 - dice
    return loss


class DiceLoss(nn.Module):
    def __init__(self,n_classes):
        super().__init__()
        self.n_classes = n_classes
        
    def one_hot_encode(self,input_tensor):
        print(input_tensor.shape,'inpt')
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor==i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list,dim=1)
        return output_tensor.float()
    
    # def forward(self,input,target,weight=None,softmax=True):
    #     if softmax:
    #         inputs = F.softmax(input,dim=1)
    #     target = self.one_hot_encode(target)
    #     if weight is None:
    #         weight = [1] * self.n_classes
    #     assert inputs.shape == target.shape,'size must match'
    #     class_wise_dice = []
    #     loss = 0.0
    #     for i in range(self.n_classes):
    #         diceloss = dice_loss(inputs[:,i], target[:,i])
    #         class_wise_dice.append(diceloss)
    #         loss += diceloss * weight[i]
    #     return loss/self.n_classes
    def forward(self,input,target,weight=None,softmax=True):
        if softmax:
            inputs = F.softmax(input,dim=1)
        print(inputs.shape, target.shape,"inputs.shape, target.shape, before")
        target = self.one_hot_encode(target)
        print(inputs.shape, target.shape,"inputs.shape, target.shape, mid")
        if weight is None:
            weight = [1] * self.n_classes
        print(inputs.shape, target.shape,"inputs.shape, target.shape, after")
        assert inputs.shape == target.shape,'size must match'
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:,i], target[:,i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        return loss/self.n_classes

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            #print("weightc for c",c,weight_c)
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight))
        if len(target.shape) == len(predict.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        #print("Weight CE:",weight)
        return wce_loss


class DiceCeLoss(nn.Module):
     #predict : output of model (i.e. no softmax)[N,C,*]
     #target : gt of img [N,1,*]
    def __init__(self,num_classes,alpha=1.0):
        '''
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        '''
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        self.celoss = WeightedCrossEntropyLoss(self.num_classes)
        
    def forward(self,predict,label):
        #predict is output of the model, i.e. without softmax [N,C,*]
        #label is not one hot encoding [N,1,*]
        
        diceloss = self.diceloss(predict,label)
        celoss = self.celoss(predict,label)
        loss = celoss + self.alpha * diceloss
        return loss
        
class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)