import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.transform import resize
import random
import xarray as xr
import os

import sys
sys.path.insert(0,'/home/huangz78/hydro/unet3d/')
from models.unet3d_model import UNet3D, ResidualUNet3D
from models.dnet import weights_init,Discriminator
from pretrain_G import G_warmup

def get_args():
    parser = argparse.ArgumentParser(description='Train GAN on hydro simulation data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=25,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=3,
                        help='Batch size', dest='b_size')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-4,
                        help='Learning rate for generator', dest='lr')
    parser.add_argument('-s','--random-seed',metavar='RS',type=float,nargs='?',default=999,
                        help='Random seed', dest='manualSeed')
    parser.add_argument('-wfid','--weight-fid',type=float,default=1,
                        help='weight for data fidelity loss in the error of G net',dest='weight_fid')
    parser.add_argument('-trd','--traintotal',type=int,default=1000,
                        help='total amount of files for training', dest='traintotal')
    parser.add_argument('-ted','--testtotal',type=int,default=100,
                        help='total amount of files for testing', dest='testtotal')
    parser.add_argument('-vf','--validate-frequency',type=int,default=50,
                        help='print every # iteration',dest='validate_every')
    parser.add_argument('-g','--gnet-path',type=str,default=None,
                        help='path to load checkpoints of generator network', dest='gpath')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    manualSeed = args.manualSeed
    #manualSeed = random.randint(1, 10000) # use if want new results
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    datapath = '/mnt/shared_b/data/hydro_simulations/data/'
    ncfiles = list([])
    for file in os.listdir(datapath):
        if file.endswith(".nc"):
            ncfiles.append(file)
    print('Total amount of files:', len(ncfiles))
    img_size = 320
    dep = 8
    traintotal  = args.traintotal
    testtotal   = args.testtotal
    
    ngpu = 0
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    ############################
    ### initialization of G network
    ############################

#     gnet = UNet3D(1,1,is_segmentation=False,final_sigmoid=False)
    gnet = ResidualUNet3D(1,1,num_levels=4,is_segmentation=False,final_sigmoid=False)
    if args.gpath is not None:
        checkpoint = torch.load(args.gpath)
        gnet.load_state_dict(checkpoint['model_state_dict'])
        print(f'G net loaded from {args.gpath} successfully!\n')
        
    gnet_cmp = ResidualUNet3D(1,1,num_levels=4,is_segmentation=False,final_sigmoid=False)
    checkpoint_cmp = torch.load('/home/huangz78/checkpoints/netG_warmup.pth')
    gnet_cmp.load_state_dict(checkpoint_cmp['model_state_dict'])
    print(f'G net for comparison loaded successfully from /home/huangz78/checkpoints/netG_warmup.pth !\n')
    
    ############################
    ### training process
    ############################
    G_warmup(gnet,gnet_cmp, lr=args.lr,traintotal=traintotal,testtotal=testtotal,dep=dep,\
              img_size=img_size,\
              num_epochs=args.epochs,b_size=args.b_size,weight_fid=args.weight_fid,\
              print_every=1,validate_every=args.validate_every,\
              ngpu=ngpu,manual_seed=args.manualSeed,save_cp=True,resize_option=False,\
              noise_mode='const_rand')