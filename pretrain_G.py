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
from models.utils import illustrate,visualization, noise_generate, Nrmse, l1, aver_mse, aver_l1


def validate(testfiles,netG,dep=8,batchsize=5,seed=0,img_size=320, \
             device="cpu",testfile_num=100,criterion=nn.L1Loss()):
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
            dyn   = np.zeros((batchsize,1,dep,256,256))
            noise = np.zeros((batchsize,1,dep,256,256))
            bfile = 0
            while bfile < batchsize:
                filename = testfiles[fileind+bfile]
                sim = xr.open_dataarray(datapath+filename)        
                for t in range(dep):
                    dyn[bfile,0,t,:,:] = resize(sim.isel(t=t)[:img_size,:img_size].values,(256,256),anti_aliasing=True)
                maxval_tmp = np.max( np.abs(dyn[bfile,0,:,:,:]).flatten() ) # normalize each File
                if maxval_tmp > normalize_factor:
                    bfile -= 1
                    fileind += 1
                else:
                    dyn[bfile,0,:,:,:] = dyn[bfile,0,:,:,:] / normalize_factor
                    noise[bfile,0,:,:,:] = noise_generate(dyn[bfile,0,:,:,:],mode='const_rand')
                sim.close()
                bfile += 1
            fileind += batchsize
            batch_step += 1

            dyn      = torch.tensor(dyn).to(torch.float); noise = torch.tensor(noise).to(torch.float)
            real_cpu = dyn.to(device)
            b_size   = real_cpu.size(0)

            ## Test with all-fake batch
            fake       = netG(noise + real_cpu)
            errG_fake  = criterion(fake, real_cpu)

            eval_score = eval_score + errG_fake
            nrmse      = nrmse      + aver_mse(fake,real_cpu)  * fake.shape[0]
            nl1err     = nl1err     + aver_l1(fake,real_cpu)   * fake.shape[0]
    
    # set the model back to training mode
    netG.train()
    return eval_score/(batch_step*batchsize), nrmse/(batch_step*batchsize), nl1err/(batch_step*batchsize) #, nlinferr/filenum

def G_warmup(netG,lr=1e-5,beta1=0.5,\
              traintotal=500,testtotal=10,num_epochs=5,\
              dep=8,b_size=5,\
              fileexp_ind=5010,\
              print_every=10,validate_every=100,\
              img_size=320,ngpu=0,manual_seed=999,device="cpu",\
              save_cp=False,make_plot=False):
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
    fileexp_ind      : the index of the fixed image file for illustrating performance of G network
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
    G_losses   = []; val_losses = [] 
    nrmse_val  = []; l1_val   = []; nrmse_train = list([]); l1_train = list([])
    
    fidelity_loss = nn.L1Loss()
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    schedulerG = StepLR(optimizerG, step_size=100, gamma=0.8)
    print("Starting Training Loop...")
    
    normalize_factor = 50
    
    fileexp  = ncfiles[fileexp_ind] # fix one file to observe the outcome of the generator
    simexp   = xr.open_dataarray(datapath + fileexp)
    dynexp   = np.zeros((1,1,dep,256,256))
    noiseexp = np.zeros((1,1,dep,256,256)) # fix one set of dynamical frames for viewing
    for t in range(dep):
        dynexp[0,0,t,:,:] = resize(simexp.isel(t=t)[:img_size,:img_size].values,(256,256),anti_aliasing=True)
#     normalize_factor   = np.max( np.abs(dynexp[0,0,:,:,:]).flatten() )
    dynexp[0,0,:,:,:]  = dynexp[0,0,:,:,:] / normalize_factor
    noiseexp = noise_generate(dynexp,mode='const_rand')
    
    dynexp   = torch.tensor(dynexp).to(torch.float).to(device)
    noiseexp = torch.tensor(noiseexp).to(torch.float).to(device)
    
    for epoch in range(num_epochs):
        try:
            fileind = 0; global_step = 0
            while fileind < traintotal:
                current_b_size = min(b_size,traintotal-fileind)
                dyn = np.zeros((current_b_size,1,dep,256,256))
                noise = np.zeros(dyn.shape)
                bfile = 0
                while (bfile < current_b_size) and (fileind+bfile < traintotal):
                # Format batch: prepare training data for G network
                    filename = trainfiles[fileind+bfile]
                    sim = xr.open_dataarray(datapath+filename)
                    for t in range(dep):
                        dyn[bfile,0,t,:,:] = resize(sim.isel(t=t)[:img_size,:img_size].values,(256,256),anti_aliasing=True)
                    maxval_tmp = np.max( np.abs(dyn[bfile,0,:,:,:]).flatten() ) # normalize each File
                    if maxval_tmp > normalize_factor:
                        bfile -= 1
                        fileind += 1
                    else:                        
                        dyn[bfile,0,:,:,:] = dyn[bfile,0,:,:,:] / normalize_factor
                        noise[bfile,0,:,:,:] = noise_generate(dyn[bfile,0,:,:,:],mode='const_rand')
                    sim.close()
                    bfile += 1
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
                errG = fidelity_loss(fake, real_cpu)
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
                if global_step%validate_every==0 :
                    val_loss,nrmse,l1err = validate(testfiles,netG,criterion=fidelity_loss,dep=dep,batchsize=3,\
                                                    seed=manual_seed,img_size=img_size,\
                                                    device=device,testfile_num=testtotal)
                    val_losses.append(val_loss.item())
                    nrmse_val.append(nrmse.item()); l1_val.append(l1err.item())
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
                        torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + 'netG_warmup.pt')
                        np.savez('/home/huangz78/checkpoints/gnet_warmup_track.npz',\
                                 g_loss=G_losses,val_loss=val_losses,\
                                 nrmse_train=nrmse_train,l1_train=l1_train,\
                                 nrmse_val=nrmse_val,l1_val=l1_val)
                        print(f'\t Checkpoint saved at epoch {epoch + 1}, iteration {global_step}!')
                global_step += 1
        except KeyboardInterrupt: # need debug
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + 'netG_warmup.pt')
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
