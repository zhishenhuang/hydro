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

real_label = 1.
fake_label = 0.
datapath = '/mnt/shared_b/data/hydro_simulations/data/'

def load_data_batch(fileind,trainfiles,b_size=5,dep=8,img_size=320,resize_option=False,noise_mode='const_rand',normalize_factor=50):
    traintotal = len(trainfiles)   
    current_b_size = min(b_size,traintotal-fileind)
    if not resize_option:
        dyn = np.zeros((current_b_size,1,dep,img_size,img_size))
    else:
        dyn = np.zeros((current_b_size,1,dep,256,256))
    noise = np.zeros(dyn.shape)
    bfile = 0
    while (bfile < current_b_size) and (fileind+bfile < traintotal):
    # Format batch: prepare training data for D network
        filename = trainfiles[fileind+bfile]
        sim = xr.open_dataarray(datapath+filename)
        for t in range(dep):
            if not resize_option:
                dyn[bfile,0,t,:,:] = sim.isel(t=t)[:img_size,:img_size].values
            else:
                dyn[bfile,0,t,:,:] = resize(sim.isel(t=t)[:img_size,:img_size].values,(256,256),anti_aliasing=True)
        maxval_tmp = np.max( np.abs(dyn[bfile,0,:,:,:]).flatten() ) # normalize each File
        if maxval_tmp > normalize_factor:
            bfile -= 1
            fileind += 1
        else:                        
            dyn[bfile,0,:,:,:] = dyn[bfile,0,:,:,:] / normalize_factor
            noise[bfile,0,:,:,:] = noise_generate(dyn[bfile,0,:,:,:],mode=noise_mode,scaling=dyn[bfile,0,:,:,:].max())
#                         for t in range(dep): # different noise for each frame when using a 'for' loop
#                             noise[bfile,0,t,:,:] = noise_generate(dyn[bfile,0,t,:,:],mode='linear')
        sim.close()
        bfile += 1
    return dyn, noise, current_b_size

def validate(testfiles,netD,netG,dep=8,batchsize=2,seed=0,img_size=320, \
             device="cpu",sigmoid_on=False,testfile_num=100,\
             resize_option=False,noise_mode='const_rand',normalize_factor=50):
#     filenum = len(testfiles)
    batchsize = min(testfile_num,batchsize)
    random.seed(seed)
    torch.manual_seed(seed)
    datapath = '/mnt/shared_b/data/hydro_simulations/data/'
    real_label = 1.
    fake_label = 0.
    
    # set the model in eval mode
    netD.eval()
    netG.eval()
    eval_score = 0; nrmse = 0; nl1err = 0; nlinferr = 0
    # evaluate on validation set
    fileind = 0
    criterion = nn.MSELoss() if sigmoid_on else nn.BCEWithLogitsLoss()
    batch_step = 0
    with torch.no_grad():
        while fileind < testfile_num:
            dyn, noise, b_size = load_data_batch(fileind,testfiles,b_size=batchsize,dep=dep,img_size=img_size,resize_option=resize_option,noise_mode=noise_mode,normalize_factor = normalize_factor)
            fileind += batchsize
            batch_step += 1

            dyn = torch.tensor(dyn).to(torch.float); noise = torch.tensor(noise).to(torch.float)
            real_cpu = dyn.to(device)
#             b_size = real_cpu.size(0)
            del dyn
            ## Test with all-real batch
            Dx1_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
#             Dx1 = netD(real_cpu).view(-1).detach()
            Dx1 = torch.sigmoid(netD(real_cpu)).view(-1).detach() if sigmoid_on else netD(real_cpu).view(-1).detach()
            # Calculate loss on all-real batch
            errD_real = criterion(Dx1, Dx1_label)

            ## Test with all-fake batch
            fake = netG(noise + real_cpu).detach()
            DGz1_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            # Classify all fake batch with D
#             DGz1 = netD(fake).view(-1).detach()
            DGz1 = torch.sigmoid(netD(fake)).view(-1).detach() if sigmoid_on else netD(fake).view(-1).detach()
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(DGz1, DGz1_label)

            eval_score = eval_score + (errD_real + errD_fake)
            nrmse      = nrmse      + aver_mse(fake,real_cpu)  * fake.shape[0]
            nl1err     = nl1err     + aver_l1(fake,real_cpu)   * fake.shape[0]
#             nlinferr   = nlinferr   + aver_linf(fake,real_cpu) * fake.shape[0]
    return eval_score/(batch_step*batchsize), nrmse/(batch_step*batchsize), nl1err/(batch_step*batchsize) #, nlinferr/filenum


def compute_mass(imgs,Rrho=1,Rz=1):
    '''
    computing through cylindrical coordinate
    input  --- imgs: [#batch, #channel, #dep, #heg, #wid]
    output --- mass: [#batch, #dep]
    '''
    drho = Rrho / imgs.shape[3]
    dz   = Rz   / imgs.shape[4]
    metrics = torch.linspace(0,Rrho,imgs.shape[4]).repeat(imgs.shape[3],1)
    integrand = imgs * metrics
    mass = 2*np.pi * torch.sum(integrand,dim=(3,4)) * drho * dz 
    return torch.squeeze(mass)

def gan_train(netG,netD,lrd=1e-5,lrg=2e-5,beta1=0.5,\
              traintotal=500,testtotal=10,num_epochs=5,\
              weight_fid=1-1e-3,dep=8,b_size=5,update_D_every=10,update_G_every=1,\
              fileexp_ind=5010,sigmoid_on=False,\
              print_every=10,validate_every=100,\
              img_size=320,ngpu=0,manual_seed=999,device="cpu",\
              save_cp=False,make_plot=False,\
              resize_option=False,noise_mode='const_rand'):
    '''
    netG             : input Generative network
    netD             : input Discriminative network
    lrd              : learning rate for D network. 
    lrg              : learning rate for G network.
    dep              : the amount of consecutive frames to include in one input
    b_size           : batch size for training
    weight_fid       : weight to put on the error in discriminating real data
    traintotal       : total amount of files for training
    testtotal        : total amount of files for testing
    num_epochs       : number of epochs to run
    
    update_D_every   : for this many global_steps, we make one update to parameters of D network
    update_G_every   : for this many global steps, we make one update to parameters of G network
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
    
    real_label = 1.
    fake_label = 0.
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    datapath = '/mnt/shared_b/data/hydro_simulations/data/'
    ncfiles = list([])
    for file in os.listdir(datapath):
        if file.endswith(".nc"):
            ncfiles.append(file)
    print('Total amount of available files:', len(ncfiles))
    print('Train file amount: {}'.format(traintotal))
    print('Test file amount:  {}'.format(testtotal))
    
    filestart  = 8000
    trainfiles = ncfiles[filestart:filestart+traintotal+800]
    testfiles  = ncfiles[filestart+traintotal+800:] # traintotal+testtotal+800
    
    print('weight of fidelity loss in errG = ', weight_fid)
    assert(weight_fid>=0)
    # Training Loop
    # Lists to keep track of progress
#     img_list = []
    G_losses  = []; D_losses = []; val_losses = [] 
    nrmse_val = []; l1_val   = []; nrmse_train = list([]); l1_train = list([])
#     criterion  = nn.BCELoss() if sigmoid_on else nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss() if sigmoid_on else nn.BCEWithLogitsLoss()
#     fidelity_loss = nn.L1Loss()
    fidelity_loss = nn.MSELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lrd, betas=(beta1, 0.999))
#     schedulerD = ReduceLROnPlateau(optimizerD, 'min',factor=0.8,patience=20)
    optimizerG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, 0.999))
#     schedulerG = StepLR(optimizerG, step_size=150, gamma=0.9)
    print("Starting Training Loop...")
    # For each epoch
#     with torch.autograd.set_detect_anomaly(True):
    
    normalize_factor = 50
        
    for epoch in range(num_epochs):
        try:
            fileind = 0; global_step = 0; D_update_ind = 0; G_update_ind = 0
            while fileind < traintotal:
                # set the model back to training mode
                netD.train()
                netG.train()
                
                dyn, noise, current_b_size = load_data_batch(fileind,trainfiles,b_size=b_size,dep=dep,img_size=img_size,resize_option=resize_option,noise_mode=noise_mode,normalize_factor=normalize_factor)
                dyn = torch.tensor(dyn).to(torch.float); noise = torch.tensor(noise).to(torch.float)
                real_cpu = dyn.to(device)
                del dyn
                fileind += b_size
    #             b_size = real_cpu.size(0)

                # Generate fake image batch with G
                fake = netG(noise + real_cpu)
                l2err_tmp = Nrmse(fake,real_cpu); l1err_tmp = l1(fake,real_cpu)
                nrmse_train.append(l2err_tmp.item()); l1_train.append(l1err_tmp.item())
                print(f"[{global_step}][{epoch+1}/{num_epochs}][{fileind}/{traintotal}]: l2err = {l2err_tmp},  l1err = {l1err_tmp}")
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ############################
                if D_update_ind%update_D_every == 0:
                    ## Train with all-real batch           
                    netD.zero_grad()
                    Dx1_label = torch.full((current_b_size,), real_label, dtype=torch.float, device=device)
                    # Forward pass real batch through D
#                     Dx1 = netD(real_cpu).view(-1)
                    Dx1 = torch.sigmoid(netD(real_cpu)).view(-1) if sigmoid_on else netD(real_cpu).view(-1)
                    # Calculate loss on all-real batch
                    errD_real = criterion(Dx1, Dx1_label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = Dx1.mean().item() 

                    ## Train with all-fake batch
                    DGz1_label = torch.full((current_b_size,), fake_label, dtype=torch.float, device=device)
        #                 label.fill_(fake_label)
                    # Classify all fake batch with D
#                     DGz1 = netD(fake.detach()).view(-1)
                    DGz1 = torch.sigmoid(netD(fake.detach())).view(-1) if sigmoid_on else netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterion(DGz1, DGz1_label)
                    # Calculate the gradients for this batch
                    errD_fake.backward(retain_graph=True)
                    D_G_z1 = DGz1.mean().item() 
                    # Add the gradients from the all-real and all-fake batches
                    errD = errD_real + errD_fake
                    # Update D
                    optimizerD.step()           
                    D_losses.append(errD.item())    
                D_update_ind += 1

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ############################
                if G_update_ind%update_G_every == 0:
                    DGz2_label = torch.full((current_b_size,), real_label, dtype=torch.float, device=device)
        #                 label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
#                     DGz2 = netD(fake).view(-1)
                    DGz2 = torch.sigmoid(netD(fake)).view(-1) if sigmoid_on else netD(fake).view(-1)
                    D_G_z2 = DGz2.mean().item()
                    # Calculate G's loss based on this output
                    mass_fake = compute_mass(fake)
                    mass_fake.retain_grad()
                    mass_real = compute_mass(real_cpu)
                    errG = criterion(DGz2, DGz2_label) + weight_fid * fidelity_loss(mass_fake, mass_real) # Jul 23, add the mass conservation term
                    # Calculate gradients for G
                    netG.zero_grad()
                    errG.backward()
                    # Update G
                    optimizerG.step()
                    # Save Losses for plotting later
                    G_losses.append(errG.item())
                G_update_ind += 1
                
                ############################
                # Output training stats, and visualization
                ############################
                if (global_step%print_every==0) and (len(D_losses)>0) :
                    print('[%d/%d][%d/%d]\t Loss_D: %.4f\t Loss_G: %.4f\t D(x): %.4f\t D(G(z)): D %.4f/0   G %.4f/1'
                          % (epoch+1, num_epochs, fileind, traintotal, \
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    np.savez('/home/huangz78/checkpoints/gan_train_track.npz',\
                                 g_loss=G_losses,d_loss=D_losses,val_loss=val_losses,\
                                 nrmse_train=nrmse_train,l1_train=l1_train,\
                                 nrmse_val=nrmse_val,l1_val=l1_val)
                global_step += 1 
#                 if global_step%validate_every==0 :
            ############################
            # Validation
            ############################
            val_loss,nrmse,l1err = validate(testfiles,netD,netG,dep=dep,batchsize=2,\
                                            seed=manual_seed,img_size=img_size,\
                                            sigmoid_on=sigmoid_on,device=device,testfile_num=testtotal,\
                                            resize_option=resize_option,noise_mode=noise_mode,normalize_factor=normalize_factor)
            val_losses.append(val_loss.item())
            nrmse_val.append(nrmse.item()); l1_val.append(l1err.item())
            print('validation loss = {}, average nrmse = {}, average l1 err = {}'.format( val_loss.item(),nrmse.item(),l1err.item() ))
#                     schedulerD.step(val_loss)
#                     schedulerG.step()
                
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                try:
                    os.mkdir(dir_checkpoint)
                    print('Created checkpoint directory')
        #                 logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + 'netD.pt')
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + 'netG.pt')
                np.savez('/home/huangz78/checkpoints/gan_train_track.npz',\
                         g_loss=G_losses,d_loss=D_losses,val_loss=val_losses,\
                         nrmse_train=nrmse_train,l1_train=l1_train,\
                         nrmse_val=nrmse_val,l1_val=l1_val)
                print(f'\t Checkpoint saved at epoch {epoch + 1}, iteration {global_step}!')
                
        except KeyboardInterrupt: # need debug
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + 'netD.pt')
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + 'netG.pt')
                np.savez('/home/huangz78/checkpoints/gan_train_track.npz',\
                                 g_loss=G_losses,d_loss=D_losses,val_loss=val_losses,\
                                 nrmse_train=nrmse_train,l1_train=l1_train,\
                                 nrmse_val=nrmse_val,l1_val=l1_val)
                print(f'\t Checkpoint saved at epoch {epoch+1}, iteration {global_step-1}!')
                print('D net, G net, and loss records are saved after key interrupt~')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
            
            # Check how the generator is doing by saving G's output on fixed_noise
#             if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
#                 with torch.no_grad():
#                     fake = netG(fixed_noise).detach().cpu()
#                 img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

