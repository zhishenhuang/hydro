import xarray as xr
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import os,sys
import random
from skimage.transform import resize
from models.unet3d_model import ResidualUNet3D 
import utils
from utils import *
import copy

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

datapath = '/mnt/DataB/hydro_simulations/data/'

ncfiles = list([])
for file in os.listdir(datapath):
    if file.endswith(".nc"):
        ncfiles.append(file)
traintotal = 1000
valtotal   = 100
testtotal  = 300
print('Total amount of available files:', len(ncfiles))
print('Train file amount: {}'.format(traintotal))
print('Val   file amount: {}'.format(valtotal))
print('Test  file amount: {}'.format(testtotal))

trainfiles = random.sample(set(ncfiles),k=traintotal)
ncfiles = set(ncfiles) - set(trainfiles)
valfiles  = random.sample(ncfiles,k=valtotal)
ncfiles = set(ncfiles) - set(valfiles)
testfiles = random.sample(ncfiles,k=testtotal)
np.savez('/mnt/DataA/checkpoints/leo/hydro/' +f'filesUsed.npz',\
                 trainfiles=trainfiles,valfiles=valfiles,testfiles=testfiles) 



device = torch.device('cuda:0')
noise_mode = 'Abel-gaussian'

## load a G net
# gnet = ResidualUNet3D(1,1,num_levels=4,is_segmentation=False,final_sigmoid=False)
# gpath = '/mnt/DataA/checkpoints/leo/hydro/netG_wg_Abel-gaussian_epoch_3.pt'
# # gpath = '/mnt/DataA/checkpoints/leo/hydro/netG_wg_Abel-gaussian_scaling_1.0_supweigtdecay_1.0_epoch_5.pt'
# checkpoint = torch.load(gpath)
# gnet.load_state_dict(checkpoint['model_state_dict'],strict=True)
# gnet.eval()
# print(f' G net is successfully loaded from {gpath}! ')
# gnet_params_num = gnet.n_params
# print('total amount of parameters in gnet: ', gnet_params_num)


# scaling = 1
# supwgt  = .97
# batchsize = 6
# postprocess = True
# massdiff, nrmse, nl1err = test_(gnet,testfiles,batchsize=batchsize,noise_mode=noise_mode,scaling=scaling,\
#                                   device=device,postprocess=postprocess)
# dir_rec = f'/home/leo/hydro/hist_{noise_mode}_{scaling}_supwgt{supwgt}_post_{postprocess}.npz'
# np.savez(dir_rec,massdiff=massdiff,nrmse=nrmse,nl1err=nl1err)
# print(f'test result saved for noise mode {noise_mode}, scaling {scaling}, supwgt {supwgt}')




# noise_mode = 'Abel-gaussian'
# scaling = 1
# supwgt  = 1
# batchsize = 6
# massdiff, nrmse, nl1err = \
# test_(gnet,testfiles,batchsize=batchsize,noise_mode=noise_mode,scaling=scaling,device=device,postprocess=False)
# dir_rec = f'/home/leo/hydro/hist_{noise_mode}_{scaling}_supwgt{supwgt}.npz'
# np.savez(dir_rec,massdiff=massdiff,nrmse=nrmse,nl1err=nl1err)
# print(f'test result saved for noise mode {noise_mode}, scaling {scaling}, supwgt {supwgt}')



scaling = 1
massdiff, nrmse, nl1err = \
                    test_baseline(testfiles,noise_mode=noise_mode,scaling=scaling,device=device,\
                                  weight_datafid=5,weight_masscon=1e2,weight_TVA=1e-4)
dir_rec = f'/home/leo/hydro/hist_{noise_mode}_{scaling}_baseline_lambda0_5_new.npz'
np.savez(dir_rec,massdiff=massdiff,nrmse=nrmse,nl1err=nl1err)
print(f'test result saved for noise mode {noise_mode}, scaling {scaling}')
