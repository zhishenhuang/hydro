import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from skimage.transform import resize
import random
import xarray as xr
import os,sys
from utils import *

real_label = 1.
fake_label = 0.
datapath = '/mnt/DataB/hydro_simulations/data/'

def validate(testfiles,netD,netG,\
             dep=8,batchsize=2,seed=0,img_size=320, \
             sigmoid_on=False,testfile_num=100,\
             noise_mode='const_rand',normalize_factor=50,\
             device=torch.device("cpu"),resize_option=False):
#     filenum = len(testfiles)
    batchsize = min(testfile_num,batchsize)
    random.seed(seed)
    torch.manual_seed(seed)
    real_label = 1.
    fake_label = 0.
    
    # set the model in eval mode
    netD.eval()
    netG.eval()
    eval_score = 0; nrmse = 0; nl1err = 0; nlinferr = 0; Mass_diff = 0
    # evaluate on validation set
    fileind = 0
    criterion = nn.MSELoss() if sigmoid_on else nn.BCEWithLogitsLoss()
    batch_step = 0
    with torch.no_grad():
        while fileind < testfile_num:
            dyn, noise = load_data_batch(fileind,testfiles,b_size=batchsize,dep=dep,img_size=img_size,\
                                        resize_option=resize_option,\
                                        noise_mode=noise_mode,normalize_factor = normalize_factor)
            fileind += batchsize
            batch_step += 1

            real_cpu = dyn.to(device)
            del dyn
            ## Test with all-real batch
            Dx1_label = torch.full((batchsize,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            Dx1 = torch.sigmoid(netD(real_cpu)).view(-1).detach() if sigmoid_on else netD(real_cpu).view(-1).detach()
            # Calculate loss on all-real batch
            errD_real = criterion(Dx1, Dx1_label)

            ## Test with all-fake batch
            fake = netG(noise + real_cpu).detach().clamp(min=0)
            fake[real_cpu==0] = 0
            DGz1_label = torch.full((batchsize,), fake_label, dtype=torch.float, device=device)
            # Classify all fake batch with D
            DGz1 = torch.sigmoid(netD(fake)).view(-1).detach() if sigmoid_on else netD(fake).view(-1).detach()
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(DGz1, DGz1_label)
            
            mass_fake = compute_mass(fake,device=device)
            mass_real = compute_mass(real_cpu,device=device)
            mass_diff = torch.divide(torch.abs(mass_fake - mass_real), mass_real).sum()
            
            eval_score = eval_score + (errD_real + errD_fake)
            nrmse      = nrmse      + aver_mse(fake,real_cpu)  * fake.shape[0]
            nl1err     = nl1err     + aver_l1(fake,real_cpu)   * fake.shape[0]
#             nlinferr   = nlinferr   + aver_linf(fake,real_cpu) * fake.shape[0]
            Mass_diff  = Mass_diff + mass_diff
            del real_cpu, fake
    return eval_score/(batch_step*batchsize), nrmse/(batch_step*batchsize), nl1err/(batch_step*batchsize), Mass_diff/(testfile_num*dep) #, nlinferr/filenum




def gan_train(netG,netD,\
              lrd=1e-5,lrg=2e-5,beta1=0.5,\
              traintotal=500,testtotal=10,\
              num_epochs=5,b_size=5,b_size_test=5,\
              weight_fid=1-1e-3,dep=8,\
              update_D_every=10,update_G_every=1,\
              fileexp_ind=5010,sigmoid_on=False,\
              print_every=10,\
              img_size=320,manual_seed=999,\
              save_cp=False,make_plot=False,resize_option=False,\
              noise_mode='const_rand',\
              ngpu=0):
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
    G_losses  = []; D_losses = []; val_losses = [] 
    nrmse_val = []; l1_val   = []; nrmse_train = list([]); l1_train = list([])
    Massdiffs = []; bce_fake = [];  bce_real = []
#     criterion  = nn.BCELoss() if sigmoid_on else nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss() if sigmoid_on else nn.BCEWithLogitsLoss()
    bce_loss  = nn.BCELoss()
    L1_loss   = nn.L1Loss()
    L2_loss   = nn.MSELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lrd, betas=(beta1, 0.999))
#     schedulerD = ReduceLROnPlateau(optimizerD, 'min',factor=0.8,patience=20)
    optimizerG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, 0.999))
#     schedulerG = StepLR(optimizerG, step_size=150, gamma=0.9)
    print("Starting Training Loop...")
    # For each epoch
    dir_checkpoint = '/mnt/DataA/checkpoints/leo/hydro/'
    normalize_factor = 50
        
    for epoch in range(num_epochs):
        try:
            fileind = 0; global_step = 0; D_update_ind = 0; G_update_ind = 0
            while fileind < traintotal:
                # set the model back to training mode
                netD.train()
                netG.train()
                
                dyn, noise = load_data_batch(fileind, trainfiles, \
                                             b_size=b_size, dep=dep, img_size=img_size,\
                                             resize_option=resize_option,\
                                             noise_mode=noise_mode, normalize_factor=normalize_factor)
                noise    = noise.to(device)
                real_cpu = dyn.to(device)
                del dyn
                fileind += b_size

                # Generate fake image batch with G
                fake = netG(noise + real_cpu).clamp(min=0)
                with torch.no_grad():
                    fake[real_cpu==0] = 0
                l2err_tmp = Nrmse(fake,real_cpu); l1err_tmp = l1(fake,real_cpu)
                nrmse_train.append(l2err_tmp.item()); l1_train.append(l1err_tmp.item())
                print(f"[{global_step}][{epoch+1}/{num_epochs}][{fileind}/{traintotal}]: l2err = {l2err_tmp},  l1err = {l1err_tmp}")
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ############################
                if D_update_ind%update_D_every == 0:
                    ## Train with all-real batch           
                    netD.zero_grad()
                    Dx1_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    # Forward pass real batch through D
                    Dx1 = torch.sigmoid(netD(real_cpu)).view(-1) if sigmoid_on else netD(real_cpu).view(-1)
                    with torch.no_grad():
                        bce_loss_real = bce_loss(Dx1,torch.ones_like(Dx1)).detach()
                        bce_real.append(bce_loss_real.item())
                    # Calculate loss on all-real batch
                    errD_real = criterion(Dx1, Dx1_label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = Dx1.mean().item() 

                    ## Train with all-fake batch
                    DGz1_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        #                 label.fill_(fake_label)
                    # Classify all fake batch with D
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
                    DGz2_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        #                 label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    DGz2 = torch.sigmoid(netD(fake)).view(-1) if sigmoid_on else netD(fake).view(-1)
                    D_G_z2 = DGz2.mean().item()
                    with torch.no_grad():
                        bce_loss_fake = bce_loss(DGz2,torch.zeros_like(DGz2)).detach()
                        bce_fake.append(bce_loss_fake.item())
                    # Calculate G's loss based on this output
                    mass_fake = compute_mass(fake)
                    mass_fake.retain_grad()
                    mass_real = compute_mass(real_cpu)
                    errG = criterion(DGz2, DGz2_label) + weight_fid * L2_loss(mass_fake, mass_real) 
                    # Calculate gradients for G
                    netG.zero_grad()
                    errG.backward()
                    # Update G
                    optimizerG.step()                    
                    G_losses.append(errG.item())
                G_update_ind += 1
                del real_cpu, fake
                ############################
                # Output training stats, and visualization
                ############################
                if (global_step%print_every==0) and (len(D_losses)>0) :
                    print(f'[{epoch+1}/{num_epochs}][{fileind}/{traintotal}]\t Loss_D: {errD.item():.4f}\t Loss_G: {errG.item():.4f}\t D(x): {D_x:.4f}\t D(G(z)): D {D_G_z1:.4f}/0   G {D_G_z2:.4f}/1')
                    np.savez(dir_checkpoint + 'gan_train_track.npz',\
                                 g_loss=G_losses,d_loss=D_losses,val_loss=val_losses,\
                                 nrmse_train=nrmse_train,l1_train=l1_train,\
                                 nrmse_val=nrmse_val,l1_val=l1_val)
                global_step += 1 
            ############################
            # Validation
            ############################
            val_loss,nrmse,l1err,massdiff = validate(testfiles,netD,netG,dep=dep,batchsize=b_size_test,\
                                                        seed=manual_seed,img_size=img_size,\
                                                        sigmoid_on=sigmoid_on,testfile_num=testtotal,\
                                                        resize_option=resize_option,\
                                                        noise_mode=noise_mode,normalize_factor=normalize_factor,\
                                                        device=device)
            val_losses.append(val_loss.item())
            nrmse_val.append(nrmse.item()); l1_val.append(l1err.item()); Massdiffs.append(massdiff.item())
            print(f'validation loss = {val_loss.item()}, average nrmse = {nrmse.item()}, average l1 err = {l1err.item()}')
#             schedulerD.step(val_loss)
#             schedulerG.step()
                
            if save_cp:                
                try:
                    os.mkdir(dir_checkpoint)
                    print('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + f'netD_{noise_mode}.pt')
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + f'netG_{noise_mode}.pt')
                np.savez(dir_checkpoint + 'gan_train_track.npz',\
                             g_loss=G_losses,d_loss=D_losses,val_loss=val_losses,\
                             nrmse_train=nrmse_train,l1_train=l1_train,\
                             nrmse_val=nrmse_val,l1_val=l1_val,\
                             bce_real=bce_real,bce_fake=bce_fake,Massdiffs=Massdiffs)
                print(f'\t Checkpoint saved at epoch {epoch + 1}, iteration {global_step}!')
                
        except KeyboardInterrupt: # need debug
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + f'netD_{noise_mode}.pt')
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + f'netG_{noise_mode}.pt')
                np.savez(dir_checkpoint + 'gan_train_track.npz',\
                                 g_loss=G_losses,d_loss=D_losses,val_loss=val_losses,\
                                 nrmse_train=nrmse_train,l1_train=l1_train,\
                                 nrmse_val=nrmse_val,l1_val=l1_val,\
                                 bce_real=bce_real,bce_fake=bce_fake,Massdiffs=Massdiffs)
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

