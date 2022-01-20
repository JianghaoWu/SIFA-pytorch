"""network for SIFA"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
import random


class InsResBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer = []
        self.layer += [nn.Conv2d(in_features, in_features, 3, 1, 1)]
        self.layer += [nn.InstanceNorm2d(in_features)]
        self.layer += [nn.ReLU(inplace=True)]
        self.layer += [nn.Conv2d(in_features, in_features, 3, 1, 1)]
        self.layer += [nn.InstanceNorm2d(in_features)]
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        res = x
        x = self.layer(x)
        out = F.relu(x + res, inplace=True)
        return out


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad, norm_type=None, p_dropout=None, do_relu=True):
        super().__init__()
        self.layer = [nn.Conv2d(c_in, c_out, k_size, stride=stride, padding=pad)]
        if p_dropout is not None:
            self.layer += [nn.Dropout(p=p_dropout)]
        if norm_type is not None:
            if norm_type == 'BN':
                self.layer += [nn.BatchNorm2d(c_out)]
            if norm_type == 'IN':
                self.layer += [nn.InstanceNorm2d(c_out)]
        if do_relu:
            self.layer += [nn.ReLU(inplace=True)]
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class InsDeconv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.layer = [nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1)]
        self.layer += [nn.InstanceNorm2d(c_out)]
        self.layer += [nn.ReLU(inplace=True)]
        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class DilateBlock(nn.Module):
    def __init__(self, c_in, c_out, p_dropout=0.25):
        super().__init__()
        self.layer = [nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=2, dilation=2)]
        self.layer += [nn.Dropout(p_dropout)]
        self.layer += [nn.BatchNorm2d(c_out)]
        self.layer += [nn.ReLU(inplace=True)]
        self.layer += [nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=2, dilation=2)]
        self.layer += [nn.Dropout(p_dropout)]
        self.layer += [nn.BatchNorm2d(c_out)]
        self.layer = nn.Sequential(*self.layer)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.layer(x))


class BNResBlock(nn.Module):
    def __init__(self, c_in, c_out, p_dropout=0.25):
        super().__init__()
        # c_in==c_out or c_out = 2*c_in
        self.c_in = c_in
        self.c_out = c_out
        p = p_dropout
        self.layer = [nn.Conv2d(c_in, c_out, 3, 1, 1)]
        self.layer += [nn.Dropout(p)]
        self.layer += [nn.BatchNorm2d(c_out)]
        self.layer += [nn.ReLU(inplace=True)]
        self.layer += [nn.Conv2d(c_out, c_out, 3, 1, 1)]
        self.layer += [nn.Dropout(p)]
        self.layer += [nn.BatchNorm2d(c_out)]
        self.layer = nn.Sequential(*self.layer)

    def expand_channels(self, x):
        # expand channels [B,C,H,W] to [B,2C,H,W]
        all0 = torch.zeros_like(x)
        z1, z2 = torch.split(all0, x.size(1) // 2, dim=1)
        return torch.cat((z1, x, z2), dim=1)

    def forward(self, x):
        if self.c_in == self.c_out:
            return F.relu(x + self.layer(x), inplace=True)
        else:
            out0 = self.layer(x)
            out1 = self.expand_channels(x)
            return F.relu(out0 + out1, inplace=True)



def get_scheduler(optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    return scheduler


class G(nn.Module):
    """G from cyclegan"""
    def __init__(self, inputdim=1, skip=True):
        super().__init__()
        self._skip = skip
        self.model = [BasicBlock(inputdim, 32, 7, 1, 3, 'IN')]
        self.model += [BasicBlock(32, 64, 3, 2, 1, 'IN')]
        self.model += [BasicBlock(64, 128, 3, 2, 1, 'IN')]
        for _ in range(9):
            self.model += [InsResBlock(128)]
        self.model += [InsDeconv(128, 64)]
        self.model += [InsDeconv(64, 32)]
        self.model += [BasicBlock(32, 1, 7, 1, 3, do_relu=False)]
        self.model = nn.Sequential(*self.model)
        self.tanh = nn.Tanh()

    def forward(self, inputgen, inputimg):
        out = self.model(inputgen)
        if self._skip is True:
            out = self.tanh(out + inputimg)
        else:
            out = self.tanh(out)
        return out


class Encoder(nn.Module):
    def __init__(self, inputdim=1, p_dropout=0.25):
        super().__init__()
        self.model = nn.Sequential(
            BasicBlock(inputdim, 16, 7, 1, 3, norm_type='BN', p_dropout=0.25),
            BNResBlock(16, 16),
            nn.MaxPool2d(kernel_size=2),
            BNResBlock(16, 32),
            nn.MaxPool2d(kernel_size=2),
            BNResBlock(32, 64),
            BNResBlock(64, 64),
            nn.MaxPool2d(kernel_size=2),
            BNResBlock(64, 128),
            BNResBlock(128, 128),
            BNResBlock(128, 256),
            BNResBlock(256, 256),
            BNResBlock(256, 256),
            BNResBlock(256, 256),
            BNResBlock(256, 512),
            BNResBlock(512, 512),
            DilateBlock(512, 512),
            DilateBlock(512, 512),
            BasicBlock(512, 512, 3, 1, 1, norm_type='BN', p_dropout=0.25),
            BasicBlock(512, 512, 3, 1, 1, norm_type='BN', p_dropout=0.25)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Decoder(nn.Module):
    def __init__(self, skip=True):
        super().__init__()
        self._skip = skip
        self.model = [BasicBlock(512, 128, 3, 1, 1, 'IN')]
        for _ in range(4):
            self.model += [InsResBlock(128)]
        self.model += [InsDeconv(128, 64)]
        self.model += [InsDeconv(64, 64)]
        self.model += [InsDeconv(64, 32)]
        self.model += [BasicBlock(32, 1, 7, 1, 3, do_relu=False)]
        self.model = nn.Sequential(*self.model)
        self.tanh = nn.Tanh()

    def forward(self, inputde, inputimg):
        out = self.model(inputde)
        if self._skip is True:
            out = self.tanh(out + inputimg)
        else:
            out = self.tanh(out)
        return out


class Seg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(BasicBlock(512, num_classes, 1, 1, 0, do_relu=False),
                                   nn.Upsample(scale_factor=8,mode='bilinear'))

    def forward(self, x):
        x = self.model(x)
        return x


class Dis(nn.Module):
    def __init__(self, input_nc=1):
        super().__init__()

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class Dis_aux(nn.Module):
    def __init__(self, input_nc=1):
        super().__init__()

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        self.share = nn.Sequential(*model)
        self.model = nn.Sequential(nn.Conv2d(512, 1, 4, padding=1))
        self.model_aux = nn.Sequential(nn.Conv2d(512, 1, 4, padding=1))
        

    def forward(self, x):
        x = self.share(x)
        x = self.model(x)
        return x
    
    def forward_aux(self,x):
        x = self.share(x)
        x = self.model_aux(x)
        return x
        

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size=50):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images

