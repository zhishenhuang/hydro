import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from skimage.transform import resize
import random
import xarray as xr
import os,sys
import matplotlib.pyplot as plt
from models.utils import *
from utils import *

def validate(testfiles,netG,dep=8,batchsize=2,seed=0,img_size=320, \
             device="cpu",testfile_num=100,\
             resize_option=False,noise_mode='const_rand',\
             normalize_factor=50,criterion=nn.L1Loss()):
#     filenum = len(testfiles)
    batchsize = min(testfile_num,batchsize)
    random.seed(seed)
    torch.manual_seed(seed)
    datapath = '/mnt/shared_b/data/hydro_simulations/data/'
    # set the model in eval mode
    netG.eval()
    eval_score = 0; nrmse = 0; nl1err = 0; nlinferr = 0
    # evaluate on validation set
    fileind = 0
    normalize_factor = 50
    batch_step = 0
    with torch.no_grad():
        while fileind < testfile_num:
            dyn, noise, _ = load_data_batch(fileind,testfiles,b_size=batchsize,dep=dep,img_size=img_size,resize_option=resize_option,noise_mode=noise_mode,normalize_factor = normalize_factor)
            fileind += batchsize
            batch_step += 1

            dyn      = torch.tensor(dyn).to(torch.float); noise = torch.tensor(noise).to(torch.float)
            real_cpu = dyn.to(device)

            ## Test with all-fake batch
            fake       = netG(noise + real_cpu).detach()
            errG_fake  = criterion(fake, real_cpu)

            eval_score = eval_score + errG_fake
            nrmse      = nrmse      + aver_mse(fake,real_cpu)  * fake.shape[0]
            nl1err     = nl1err     + aver_l1(fake,real_cpu)   * fake.shape[0]
    
    # set the model back to training mode
    netG.train()
    return eval_score/(batch_step*batchsize), nrmse/(batch_step*batchsize), nl1err/(batch_step*batchsize) #, nlinferr/filenum

def G_warmup(netG,netG_cmp,lr=1e-5,beta1=0.5,\
              traintotal=500,testtotal=10,num_epochs=5,\
              dep=8,b_size=5,weight_fid=10,\
              print_every=10,validate_every=100,\
              img_size=320,ngpu=0,manual_seed=999,device="cpu",\
              save_cp=False,make_plot=False,\
              noise_mode='const_rand',resize_option = False):
    '''
    netG             : input Generative network
    lr               : learning rate for G network
    dep              : the amount of consecutive frames to include in one input
    b_size           : batch size for training
    traintotal       : total amount of files for training
    testtotal        : total amount of files for testing
    num_epochs       : number of epochs to run
    beta1            : hyperparameter for the optimizer
    
    save_cp          : whether to save models
    print_every      : print every this many iterations
    make_plot        : whether to show learning loss curves and plot sample denoised frames
    sigmoid_on       : whether to add a sigmoid layer to the output of D network
    
    img_size         : clip this size from the original image files in the hydro data, default value 320
    ngpu             : number of GPUs
    manual_seed      : random seed for reproducing the evaluation
    
    '''
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    datapath = '/mnt/shared_b/data/hydro_simulations/data/'
    ncfiles = list([])
    for file in os.listdir(datapath):
        if file.endswith(".nc"):
            ncfiles.append(file)
    print('Total amount of available files:', len(ncfiles))
    print('Train file amount: {}'.format(traintotal))
    print('Test file amount:  {}'.format(testtotal))
    
    filestart  = 6000
    trainfiles = ncfiles[filestart:filestart+traintotal+800]
    testfiles  = ncfiles[filestart+traintotal+800:] # traintotal+testtotal+800
    
    # Training Loop
    # Lists to keep track of progress
#     img_list = []
    G_losses   = []; val_losses = []; val_losses_cmp = [] 
    nrmse_val  = []; l1_val   = []
    nrmse_train = list([]); l1_train = list([])
    nrmse_val_cmp = []; l1_val_cmp = []
    
    fidelity_loss = nn.L1Loss()
#     fidelity_loss = nn.MSELoss()
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    schedulerG = StepLR(optimizerG, step_size=100, gamma=0.9)
    print("Starting Training Loop...")
    
    normalize_factor = 50
    for epoch in range(num_epochs):
        try:
            fileind = 0; global_step = 0
            while fileind < traintotal:
                dyn, noise, _ = load_data_batch(fileind,trainfiles,b_size=b_size,dep=dep,img_size=img_size,resize_option=resize_option,noise_mode=noise_mode,normalize_factor=normalize_factor)
                dyn      = torch.tensor(dyn).to(torch.float); noise = torch.tensor(noise).to(torch.float)
                real_cpu = dyn.to(device)
                fileind += b_size

                # Generate fake image batch with G
                fake = netG(noise + real_cpu)
                l2err_tmp = Nrmse(fake,real_cpu); l1err_tmp = l1(fake,real_cpu)
                nrmse_train.append(l2err_tmp.item()); l1_train.append(l1err_tmp.item())
                print(f"[{global_step}][{epoch+1}/{num_epochs}][{fileind}/{traintotal}]: l2err = {l2err_tmp},  l1err = {l1err_tmp}")
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ############################                    
                # Calculate G's loss based on this output
                mass_fake = compute_mass(fake)
                mass_fake.retain_grad()
                mass_real = compute_mass(real_cpu)
                errG = fidelity_loss(fake, real_cpu) + weight_fid * fidelity_loss(mass_fake, mass_real)
                # Calculate gradients for G
                netG.zero_grad()
                errG.backward()
                # Update G
                optimizerG.step()
                # Save Losses for plotting later
                G_losses.append(errG.item())              
                
                ############################
                # Output training stats, and visualization
                ############################
                if (global_step%print_every==0):
                    print('[%d/%d][%d/%d]\t  Loss_G: %.4f\t ' % (epoch+1, num_epochs, fileind, traintotal, errG.item()))
                    np.savez('/home/huangz78/checkpoints/gnet_warmup_track.npz',\
                                 g_loss=G_losses,val_loss=val_losses,\
                                 nrmse_train=nrmse_train,l1_train=l1_train,\
                                 nrmse_val=nrmse_val,l1_val=l1_val)
                global_step += 1
           ############################
           # Validation
           ############################
            val_loss,nrmse,l1err = validate(testfiles,netG,dep=dep,batchsize=2,\
                                            seed=manual_seed,img_size=img_size,\
                                            device=device,testfile_num=testtotal,\
                                            resize_option=resize_option,noise_mode=noise_mode,\
                                            normalize_factor=normalize_factor,\
                                            criterion=fidelity_loss)
            val_loss_cmp,nrmse_cmp,l1err_cmp = validate(testfiles,netG_cmp,dep=dep,batchsize=2,\
                                            seed=manual_seed,img_size=img_size,\
                                            device=device,testfile_num=testtotal,\
                                            resize_option=resize_option,noise_mode=noise_mode,\
                                            normalize_factor=normalize_factor,\
                                            criterion=fidelity_loss)
            val_losses.append(val_loss.item())
            nrmse_val.append(nrmse.item()); l1_val.append(l1err.item())
            val_losses_cmp.append(val_loss_cmp.item())
            nrmse_val_cmp.append(nrmse_cmp.item()); l1_val_cmp.append(l1err_cmp.item())
            print('validation loss = {}, average nrmse = {}, average l1 err = {}'.format( val_loss.item(),nrmse.item(),l1err.item() ))      
            schedulerG.step()
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                try:
                    os.mkdir(dir_checkpoint)
                    print('Created checkpoint directory')
        #                 logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + 'netG_warmup_masspenned.pt')
                np.savez('/home/huangz78/checkpoints/gnet_warmup_track.npz',\
                         g_loss=G_losses,val_loss=val_losses,val_loss_cmp=val_losses_cmp,\
                         nrmse_train=nrmse_train,l1_train=l1_train,\
                         nrmse_val=nrmse_val,l1_val=l1_val,\
                         nrmse_val_cmp=nrmse_val_cmp,l1_val_cmp=l1_val_cmp)
                print(f'\t Checkpoint saved at epoch {epoch + 1}, iteration {global_step}!')
                
        except KeyboardInterrupt: # need debug
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + 'netG_warmup_masspenned.pt')
                np.savez('/home/huangz78/checkpoints/gnet_warmup_track.npz',\
                                 g_loss=G_losses,val_loss=val_losses,\
                                 nrmse_train=nrmse_train,l1_train=l1_train,\
                                 nrmse_val=nrmse_val,l1_val=l1_val)
                print(f'\t Checkpoint saved at epoch {epoch+1}, iteration {global_step-1}!')
                print('G net and loss records are saved after key interrupt~')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
