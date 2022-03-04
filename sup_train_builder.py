import numpy as np
import argparse
import torch
import torch.nn as nn
import random
import os

import sys
sys.path.insert(0,'/home/leo/hydro/unet3d/')
from models.unet3d_model import UNet3D, ResidualUNet3D
from sup_train import sup_trainer

import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Train GAN on hydro simulation data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=3,
                        help='Batch size', dest='b_size')
    parser.add_argument('-bt', '--batch-size-test', metavar='BT', type=int, nargs='?', default=3,
                        help='Batch size validation', dest='b_size_test')
    parser.add_argument('-lrg', '--learning-rate-g', metavar='LRG', type=float, nargs='?', default=2e-6,
                        help='Learning rate for generator', dest='lrg')
    
    parser.add_argument('-s','--random-seed',metavar='RS',type=float,nargs='?',default=999,
                        help='Random seed', dest='manualSeed')
    
    parser.add_argument('-trd','--traintotal',type=int,default=1000,
                        help='total amount of files for training', dest='traintotal')
    parser.add_argument('-ted','--testtotal',type=int,default=100,
                        help='total amount of files for testing', dest='testtotal')
    
    parser.add_argument('-es','--epoch-start',type=int,default=0,
                        help='starting epoch', dest='epoch_start')
    
    parser.add_argument('-wmasscon','--weight-masscon',type=float,default=10,
                        help='weight for mass conservation term in the error of G net',dest='weight_masscon')
    
    parser.add_argument('-scaling','--scaling-noise',type=float,default=1,
                        help='scaling of noise in Abel domain',dest='scaling')
    
    parser.add_argument('-gp','--gnet-path',type=str,default=None,
                        help='path to load checkpoints of generator network', dest='gpath')
    parser.add_argument('-hp','--hist-path',type=str,default=None,
                        help='path to load training history record',dest='hpath')
    
    parser.add_argument('-glevels','--gnet-levels',type=int,default=4,
                        help='number of levels in G net',dest='g_levels')
       
    parser.add_argument('-nm', '--noise-mode', type=str, default="Abel-gaussian",
                        help='noise mode', dest='noise_mode')
    parser.add_argument('-ngpu', '--num-gpu', type=int, default=1,
                        help='number of GPUs', dest='ngpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    manualSeed = args.manualSeed
    #manualSeed = random.randint(1, 10000) # use if want new results
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    datapath = '/mnt/DataB/hydro_simulations/data/'
    ncfiles = list([])
    for file in os.listdir(datapath):
        if file.endswith(".nc"):
            ncfiles.append(file)
    print('Total amount of files:', len(ncfiles))
    img_size = 320
    resize_option = False
    dep = 8
    normalize_factor = 50
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    
    ############################
    ### initialization of two networks
    ############################
    
    gnet = ResidualUNet3D(1,1,num_levels=args.g_levels,is_segmentation=False,final_sigmoid=False)
    if args.gpath is not None:
        checkpoint = torch.load(args.gpath)
        gnet.load_state_dict(checkpoint['model_state_dict'],strict=True)
        print(f'G net loaded from {args.gpath} successfully!\n')
    else:
        print('G net is randomly initialized!')
    
    ############################
    ### training process
    ############################
    dir_checkpoint = '/mnt/DataA/checkpoints/leo/hydro/'
    sup_Trainer = sup_trainer(gnet,dep=dep,img_size=img_size,manual_seed=args.manualSeed,\
                                resize_option=resize_option,noise_mode=args.noise_mode,\
                                normalize_factor=normalize_factor,ngpu=args.ngpu,\
                                datapath=datapath,dir_checkpoint=dir_checkpoint,dir_hist=args.hpath,\
                                sigma=2,volatility=.05,xi=.02,scaling=args.scaling,white_noise_ratio=1e-4)
    
    sup_Trainer.run(lrg=args.lrg,\
                    traintotal=args.traintotal,testtotal=args.testtotal,\
                    num_epochs=args.epochs,b_size=args.b_size,b_size_test=args.b_size_test,\
                    weight_masscon=args.weight_masscon,\
                    print_every=10,epoch_start=args.epoch_start,\
                    make_plot=False,\
                    save_cp=True)