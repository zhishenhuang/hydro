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
from train import gan_train

def get_args():
    parser = argparse.ArgumentParser(description='Train GAN on hydro simulation data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='b_size')
    parser.add_argument('-lrd', '--learning-rate-d', metavar='LRD', type=float, nargs='?', default=1e-4,
                        help='Learning rate for discriminator', dest='lrd')
    parser.add_argument('-lrg', '--learning-rate-g', metavar='LRG', type=float, nargs='?', default=1e-4,
                        help='Learning rate for generator', dest='lrg')
    parser.add_argument('-s','--random-seed',metavar='RS',type=float,nargs='?',default=999,
                        help='Random seed', dest='manualSeed')
    parser.add_argument('-trd','--traintotal',type=int,default=100,
                        help='total amount of files for training', dest='traintotal')
    parser.add_argument('-ted','--testtotal',type=int,default=10,
                        help='total amount of files for testing', dest='testtotal')
    parser.add_argument('-vf','--validate-frequency',type=int,default=50,
                        help='print every # iteration',dest='validate_every')
    parser.add_argument('-wfid','--weight-fid',type=float,default=10,
                        help='weight for data fidelity loss in the error of G net',dest='weight_fid')
    parser.add_argument('-g','--gnet-path',type=str,default='/home/huangz78/checkpoints/netG_warmup.pth',
                        help='path to load checkpoints of generator network', dest='gpath')
    parser.add_argument('-d','--dnet-path',type=str,default=None,
                        help='path to laod checkpoints of discriminator network',dest='dpath')
    parser.add_argument('-dup','--update-d-every',type=int,default=10,
                        help='update d every # steps',dest='dup')
    parser.add_argument('-gup','--update-g-every',type=int,default=1,
                        help='update g every # steps',dest='gup')
    parser.add_argument('-sigmoid','--sigmoid-on',type=str,default='True',
                        help='sigmoid on',dest='sigmoid_on')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    manualSeed = args.manualSeed
    #manualSeed = random.randint(1, 10000) # use if want new results
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    if args.sigmoid_on == 'True':
        args.sigmoid_on = True
    elif args.sigmoid_on == 'False':
        args.sigmoid_on = False
    
    datapath = '/mnt/shared_b/data/hydro_simulations/data/'
    ncfiles = list([])
    for file in os.listdir(datapath):
        if file.endswith(".nc"):
            ncfiles.append(file)
    print('Total amount of files:', len(ncfiles))
    img_size = 320
    resize_option = False
    dep = 8
    traintotal  = args.traintotal
    testtotal   = args.testtotal
    
    ngpu = 0
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    ############################
    ### initialization of two networks
    ############################
#     dnet = Discriminator(ngpu,sigmoid_on=False).to(device)    
    dnet = Discriminator(ngpu,ndf=8,sigmoid_on=False,imgsize=(dep,img_size,img_size)).to(device)
    if args.dpath is None:
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        dnet.apply(weights_init)
    else:
        checkpoint = torch.load(args.dpath)
        dnet.load_state_dict(checkpoint['model_state_dict'])
        print(f'D net loaded from {args.dpath} successfully!\n')    
#     gnet = UNet3D(1,1,is_segmentation=False,final_sigmoid=False)
    gnet = ResidualUNet3D(1,1,num_levels=4,is_segmentation=False,final_sigmoid=False)
    if args.gpath is not None:
        checkpoint = torch.load(args.gpath)
        gnet.load_state_dict(checkpoint['model_state_dict'],strict=True)
        print(f'G net loaded from {args.gpath} successfully!\n')
    
    ############################
    ### training process
    ############################
    gan_train(gnet,dnet,lrd=args.lrd,lrg=args.lrg,traintotal=traintotal,testtotal=testtotal,dep=dep,\
              update_D_every=args.dup,update_G_every=args.gup,\
              img_size=img_size,\
              num_epochs=args.epochs,b_size=args.b_size,\
              weight_fid=args.weight_fid,\
              print_every=10,validate_every=args.validate_every,\
              make_plot=False,\
              sigmoid_on=args.sigmoid_on,\
              ngpu=ngpu,\
              manual_seed=args.manualSeed,\
              save_cp=True,resize_option=resize_option)