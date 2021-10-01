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
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

datapath = '/mnt/DataB/hydro_simulations/data/'

def _gradient_penalty(netD, real_data, fake_data, gp_weight=10,use_cuda=False):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = netD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
#         losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return gp_weight * ((gradients_norm - 1) ** 2).mean()

def validate(testfiles,netD,netG,dep=8,batchsize=2,seed=0,img_size=320, \
             testfile_num=100,\
             resize_option=False,noise_mode='const_rand',normalize_factor=50,\
             device="cpu"):
#     filenum = len(testfiles)
    batchsize = min(testfile_num,batchsize)
    random.seed(seed)
    torch.manual_seed(seed)
    datapath = '/mnt/DataB/hydro_simulations/data/'
    
    # set the model in eval mode
    netD.eval()
    netG.eval()
    Mass_diff = 0; nrmse = 0; nl1err = 0
    # evaluate on validation set
    fileind = 0
    batch_step = 0
    with torch.no_grad():
        while fileind < testfile_num:
            dyn, noise = load_data_batch(fileind,testfiles,b_size=batchsize,dep=dep,img_size=img_size,\
                                        resize_option=resize_option,\
                                        noise_mode=noise_mode,normalize_factor = normalize_factor)
            fileind += batchsize
            batch_step += 1
            
            real_cpu = dyn.to(device)
            noise    = noise.to(device)
            fake = netG(noise + real_cpu).clamp(min=0).detach()
            fake[real_cpu==0] = 0

            mass_fake = compute_mass(fake,device=device)
            mass_real = compute_mass(real_cpu,device=device)
            mass_diff = torch.divide(torch.abs(mass_fake - mass_real), mass_real).sum()
            
            Mass_diff  = Mass_diff + mass_diff
            nrmse      = nrmse     + aver_mse(fake,real_cpu)  * fake.shape[0]
            nl1err     = nl1err    + aver_l1(fake,real_cpu)   * fake.shape[0]
            del dyn, real_cpu, noise
            
    return Mass_diff/(testfile_num*dep), nrmse/testfile_num, nl1err/testfile_num


def wgan_train(netG,netD,\
              lrd=1e-5,lrg=2e-5,b_size=5,b_size_test=5,beta1=0.9,\
              traintotal=500,testtotal=10,num_epochs=5,\
              weight_super=.99,weight_masscon=5,delta=0.2,weight_gradpen=5,\
              dep=8,\
              update_D_every=10,update_G_every=1,\
              print_every=10,\
              img_size=320,manual_seed=999,\
              save_cp=False,make_plot=False,\
              resize_option=False,noise_mode='const_rand',\
              ngpu=0):
    '''
    netG             : input Generative network
    netD             : input Discriminative network
    lrd              : learning rate for D network. 
    lrg              : learning rate for G network.
    dep              : the amount of consecutive frames to include in one input
    b_size           : batch size for training
    weight_super     : weight on the supervised error
    weight_masscon   : weight on the error in mass conservation
    delta            : weight on the relative L2 loss of data fidelity in the W-GAN G loss
    weight_gradpen   : weight of gradient penalty in the W-GAN D loss
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
    
    img_size         : clip this size from the original image files in the hydro data, default value 320
    ngpu             : number of GPUs
    manual_seed      : random seed for reproducing the evaluation
    '''
    
    real_label = 1.
    fake_label = 0.
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    use_cuda = True if (torch.cuda.is_available() and ngpu > 0) else False
    datapath = '/mnt/DataB/hydro_simulations/data/'
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
    
    print('weight of mass conservation term in errG = ', weight_masscon)
    print('weight of supervised loss term in errG = ',   weight_super)
    assert(weight_masscon>=0)
    assert(weight_super>=0)
    # Training Loop
    # Lists to keep track of progress
    G_losses  = []; D_losses = []; nrmse_train = list([]); l1_train = list([])
    Massdiffs = []; nrmse_val = []; l1_val = []
    bce_fake = [];  bce_real = []
    
    bce_loss   = nn.BCELoss()
    L1_loss    = nn.L1Loss()
    L2_loss    = nn.MSELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lrd, betas=(beta1, 0.999))
    schedulerD = ReduceLROnPlateau(optimizerD, 'min',factor=0.8,patience=20,min_lr=1e-6)
    optimizerG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, 0.999))
    schedulerG = StepLR(optimizerG, step_size=40, gamma=0.9)
    print("Starting Training Loop...")
    
    dir_checkpoint = '/mnt/DataA/checkpoints/leo/hydro/'
    normalize_factor = 50
    global_step = 0;
    for epoch in range(num_epochs):
        try:
            fileind = 0; D_update_ind = 0; G_update_ind = 0
            bceR = 0; bceF = 0
            while fileind < traintotal:
                # set the model back to training mode
                netD.train()
                netG.train()
                
                dyn, noise = load_data_batch(fileind, trainfiles, \
                                             b_size=b_size, dep=dep, img_size=img_size,\
                                             resize_option=resize_option,\
                                             noise_mode=noise_mode, normalize_factor=normalize_factor)
                noise = noise.to(device)
                real_cpu = dyn.to(device)
                fileind += b_size
                
                # Generate fake image batch with G
                fake = netG(noise + real_cpu).clamp(min=0)
                with torch.no_grad():
                    fake[real_cpu==0] = 0
                
                l2err_tmp = aver_mse(fake,real_cpu); l1err_tmp = aver_l1(fake,real_cpu)
                nrmse_train.append(l2err_tmp.item()); l1_train.append(l1err_tmp.item())
                print(f"[{global_step+1}][{epoch+1}/{num_epochs}][{fileind}/{traintotal}]: l2err = {l2err_tmp.item()},  l1err = {l1err_tmp.item()}")
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ############################
                if D_update_ind%update_D_every == 0:
                    ## Train with all-real batch           
                    optimizerD.zero_grad()
                    # Forward pass real batch through D
                    D_real_1 = netD(real_cpu).view(-1)
                    D_fake_1 = netD(fake.detach()).view(-1)
                    gradient_penalty = _gradient_penalty(netD,real_cpu,fake,use_cuda=use_cuda,\
                                                         gp_weight=weight_gradpen)
                    
                    d_loss = D_fake_1.mean() - D_real_1.mean() + gradient_penalty
                    d_loss.backward()
                    
                    optimizerD.step()           
                    D_losses.append(d_loss.item())
                    with torch.no_grad():
                        bce_loss_real = bce_loss(D_real_1,torch.ones_like(D_real_1)).detach()
                        bceR += bce_loss_real
                        bce_real.append(bce_loss_real.item())
                D_update_ind += 1

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ############################
                if G_update_ind%update_G_every == 0:
                    
                    D_real_2 = netD(real_cpu).view(-1)
                    D_fake_2 = netD(fake).view(-1)
                    gradient_penalty = _gradient_penalty(netD,real_cpu,fake,use_cuda=use_cuda,\
                                                         gp_weight=weight_gradpen)
                   
                    d_loss = D_fake_2.mean() - D_real_2.mean() + gradient_penalty

                    mass_fake = compute_mass(fake,device=device)
                    mass_fake.retain_grad()
                    mass_real = compute_mass(real_cpu,device=device)
                    g_loss = -(1-weight_super)*d_loss + weight_super*L1(fake,real_cpu) + delta*weight_super*L2(fake,real_cpu) + weight_masscon*L2_loss(mass_fake,mass_real)
                    
                    # Calculate gradients for G
                    optimizerG.zero_grad()
                    g_loss.backward()
                    # Update G
                    optimizerG.step()
                    # Save Losses for plotting later
                    G_losses.append(g_loss.item())
                    with torch.no_grad():
                        bce_loss_fake = bce_loss(D_fake_2,torch.zeros_like(D_fake_2)).detach()
                        bceF += bce_loss_fake
                        bce_fake.append(bce_loss_fake.item())
                G_update_ind += 1
                del dyn, real_cpu, noise
                ############################
                # Validation
                ############################
                if (b_size>abs(traintotal//2-fileind)) or  (fileind > traintotal-b_size) or ((epoch==0) and (global_step==0)):
                    massdiff,nrmse,l1err = validate(testfiles,netD,netG,dep=dep,batchsize=b_size_test,\
                                                    seed=manual_seed,img_size=img_size,\
                                                    testfile_num=testtotal,\
                                                    resize_option=resize_option,\
                                                    noise_mode=noise_mode,normalize_factor=normalize_factor,\
                                                    device=device)
                    Massdiffs.append(massdiff.item())
                    nrmse_val.append(nrmse.item())
                    l1_val.append(l1err.item())
                    print('global step: {} validation result \n mass difference = {}, average nrmse = {}, average l1 err = {}'.format(global_step+1, massdiff.item(),nrmse.item(),l1err.item() ))
                    schedulerD.step(bceR+bceF)
                    bceR = 0; bceF = 0
                    schedulerG.step()
                ############################
                # Output training stats, and visualization
                ############################
                if (global_step%print_every==0) and (len(D_losses)>0) :
                    print(f'[{global_step+1}][{epoch+1}/{num_epochs}][{fileind}/{traintotal}]\t Loss_D: {d_loss.item()}\t Loss_G: {g_loss.item()}\t D(x): {D_real_2.mean().item()}\t D(G(z)): D {D_fake_1.mean().item()}/0   G {D_fake_2.mean().item()}/1')
                    if save_cp:
                        try:
                            os.mkdir(dir_checkpoint)
                            print('Created checkpoint directory')
                        except OSError:
                            pass
                        torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + f'netD_wg_{noise_mode}.pt')
                        torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + f'netG_wg_{noise_mode}.pt')
                        np.savez(dir_checkpoint +f'wgan_train_track_{noise_mode}.npz',\
                                         g_loss=G_losses,d_loss=D_losses,Massdiffs=Massdiffs,\
                                         nrmse_train=nrmse_train,l1_train=l1_train,\
                                         nrmse_val=nrmse_val,l1_val=l1_val,\
                                         bce_real=bce_real,bce_fake=bce_fake)
                        print(f'\t Checkpoint saved at epoch {epoch+1}, iteration {global_step+1}!')
                global_step += 1
            weight_super *= .97
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    print('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + f'netD_wg_{noise_mode}.pt')
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + f'netG_wg_{noise_mode}.pt')
                np.savez(dir_checkpoint +f'wgan_train_track_{noise_mode}.npz',\
                                 g_loss=G_losses,d_loss=D_losses,Massdiffs=Massdiffs,\
                                 nrmse_train=nrmse_train,l1_train=l1_train,\
                                 nrmse_val=nrmse_val,l1_val=l1_val,\
                                 bce_real=bce_real,bce_fake=bce_fake)
                print(f'\t Checkpoint saved at epoch {epoch+1}, iteration {global_step}!')
        except KeyboardInterrupt: # need debug
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + f'netD_wg_{noise_mode}.pt')
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + f'netG_wg_{noise_mode}.pt')
                np.savez(dir_checkpoint +f'wgan_train_track_{noise_mode}.npz',\
                                     g_loss=G_losses,d_loss=D_losses,Massdiffs=Massdiffs,\
                                     nrmse_train=nrmse_train,l1_train=l1_train,\
                                     nrmse_val=nrmse_val,l1_val=l1_val,\
                                     bce_real=bce_real,bce_fake=bce_fake)
                print(f'\t Checkpoint saved at epoch {epoch+1}, iteration {global_step}!')
                print('D net, G net, and loss records are saved after key interrupt~')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

