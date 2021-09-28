import torch
import torch.nn as nn
import torch.optim as optim
import random

def outsize(din,hin,win,ker=(3,3,3),stride=(1,1,1),padding=(1,1,1),dilation=(1,1,1)):
    if isinstance(ker,int):
        ker=(ker,ker,ker)
    if isinstance(stride,int):
        stride=(stride,stride,stride)
    if isinstance(padding,int):
        padding=(padding,padding,padding)
    if isinstance(dilation,int):
        dilation=(dilation,dilation,dilation)
    dout = int( (din + 2*padding[0]- dilation[0]*(ker[0]-1)-1)/stride[0] + 1  )
    hout = int( (hin + 2*padding[1]- dilation[1]*(ker[1]-1)-1)/stride[1] + 1  )
    wout = int( (win + 2*padding[2]- dilation[2]*(ker[2]-1)-1)/stride[2] + 1  )
    return dout,hout,wout

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
#     print(m)
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, nc=1,ndf=4,imgsize=(8,256,256),sigmoid_on=False):
        super(Discriminator, self).__init__()
        self.nc   = nc
        self.ndf  = ndf
        self.Dep  = imgsize[0]
        self.Heg  = imgsize[1]
        self.Wid  = imgsize[2]
        self.middep, self.midheg,self.midwid = \
        outsize(*outsize(*outsize(*outsize(*outsize(*outsize(self.Dep,self.Heg,self.Wid,\
                                                             ker=(3,4,4),stride=(1,2,2)),\
                                                    ker=(3,4,4),stride=(1,2,2)),\
                                           ker=(4,4,4),stride=(2,2,2)),\
                                  ker=(3,4,4),stride=(1,2,2)),\
                         ker=(4,4,4),stride=(2,2,2)),\
                ker=(3,4,4),stride=(1,2,2))
        self.veclen = self.ndf * 32 * self.middep * self.midheg * self.midwid 
        lwid1 = self.veclen*7//11
        lwid2 = self.veclen*3//11
        lwid3 = self.veclen*1//11
        lwid4 = 1
        
        self.main = nn.Sequential(
            # input is (self.nc) x 256 x 256
            nn.Conv3d(self.nc     , self.ndf,       (3,4,4), (1,2,2), (1,1,1), bias=False),
            nn.InstanceNorm3d(self.ndf * 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf) x 128 x 128
            nn.Conv3d(self.ndf    , self.ndf * 2,   (3,4,4), (1,2,2), (1,1,1), bias=False),
            nn.InstanceNorm3d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*2) x 64 x 64
            nn.Conv3d(self.ndf * 2, self.ndf * 4,   (4,4,4), (2,2,2), (1,1,1), bias=False),
            nn.InstanceNorm3d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*4) x 32 x 32
            nn.Conv3d(self.ndf * 4, self.ndf * 8,   (3,4,4), (1,2,2), (1,1,1), bias=False),
            nn.InstanceNorm3d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*8) x 16 x 16
            nn.Conv3d(self.ndf * 8, self.ndf * 16,  (4,4,4), (2,2,2), (1,1,1), bias=False),
            nn.InstanceNorm3d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (self.ndf*16) x 8 x 8
            nn.Conv3d(self.ndf * 16, self.ndf * 32, (3,4,4), (1,2,2), (1,1,1), bias=False),
            nn.InstanceNorm3d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),                   
            ## state size. (self.ndf*32) x 4 x 4
#             nn.Conv3d(self.ndf * 32, 1, (4,4,4), (2,1,1), (1,0,0), bias=False),
        )
        self.dense = nn.Sequential(            
            nn.Linear(self.veclen, lwid1),
            nn.LeakyReLU(),
            nn.Linear(lwid1, lwid2),
            nn.LeakyReLU(),
            nn.Linear(lwid2, lwid3),
            nn.LeakyReLU(),
            nn.Linear(lwid3, lwid4),
        )
        self.sigmoid_on = sigmoid_on
        
    @property
    def n_params(self):
        params_num = sum(p.numel() for p in self.parameters() if p.requires_grad) 
        return params_num
    
    def forward(self, input):
        batchsize = input.shape[0]
        if self.sigmoid_on:
            x_intermediate = self.main(input)
            return torch.sigmoid(self.dense(x_intermediate.view(batchsize,-1)))
        else:
            x_intermediate = self.main(input)
            return self.dense(x_intermediate.view(batchsize,-1))