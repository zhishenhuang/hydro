import torch
import torch.nn as nn
import torch.optim as optim
import random


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    print(m)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, ngpu,nc=1,ndf=64,sigmoid_on=False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc   = nc
        self.ndf  = ndf
        self.main = nn.Sequential(
            # input is (self.nc) x 256 x 256
            nn.Conv3d(self.nc     , self.ndf,       (3,4,4), (1,2,2), (1,1,1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf) x 128 x 128
            nn.Conv3d(self.ndf    , self.ndf * 2,   (3,4,4), (1,2,2), (1,1,1), bias=False),
            nn.BatchNorm3d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*2) x 64 x 64
            nn.Conv3d(self.ndf * 2, self.ndf * 4,   (4,4,4), (2,2,2), (1,1,1), bias=False),
            nn.BatchNorm3d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*4) x 32 x 32
            nn.Conv3d(self.ndf * 4, self.ndf * 8,   (3,4,4), (1,2,2), (1,1,1), bias=False),
            nn.BatchNorm3d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*8) x 16 x 16
            nn.Conv3d(self.ndf * 8, self.ndf * 16,  (4,4,4), (2,2,2), (1,1,1), bias=False),
            nn.BatchNorm3d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*16) x 8 x 8
            nn.Conv3d(self.ndf * 16, self.ndf * 32, (3,4,4), (1,2,2), (1,1,1), bias=False),
            nn.BatchNorm3d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*32) x 4 x 4
            nn.Conv3d(self.ndf * 32, 1, (4,4,4), (2,1,1), (1,0,0), bias=False),
            ### optional for Residual3dUnet
#             nn.Conv3d(self.ndf * 32, self.ndf * 64, (4,4,4), (2,2,2), (1,1,1), bias=False),
#             nn.BatchNorm3d(self.ndf * 64),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (self.ndf*32) x 4 x 4
#             nn.Conv3d(self.ndf * 64, 1       , (4,4,4), (2,2,2), (1,1,1), bias=False),
#             nn.Sigmoid()
        )
        self.sigmoid_on = sigmoid_on

    def forward(self, input):
        if self.sigmoid_on:
            return torch.sigmoid(self.main(input))
        else:
            return self.main(input)