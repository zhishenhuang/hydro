import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from skimage.transform import resize
import random
import xarray as xr
import os,sys
from models.utils import validate,illustrate,visualization

def noise_generate(frame, fl=0.006, fr=0.0179, cl=0.06,cr=0.21, sigma=1e-1 ,mode='linear' ):
    if mode == 'linear':
        factor = np.random.uniform(fl,fr,1)
        const  = np.random.uniform(cl,cr,1)
        noise  = frame * factor + const
    elif mode == 'gaussian':
        noise_mag = sigma*np.max(np.abs(frame.flatten()))
        noise = noise_mag*torch.randn(frame.shape[0],frame.shape[1])
    return noise

def gan_train(netG,netD,lrd=1e-5,lrg=2e-5,beta1=0.5,traintotal=500,testtotal=10,num_epochs=5,\
              weight_DD=1-1e-3,dep=8,b_size=5,update_D_every=10,\
              save_cp=False,make_plot=True,fileexp_ind=5010,sigmoid_on=False,\
              print_every=10,validate_every=100,\
              img_size=320,ngpu=0,manual_seed=999,device="cpu"):
    '''
    netG             : input Generative network
    netD             : input Discriminative network
    lrd              : learning rate for D network. 
    lrg              : learning rate for G network.
    dep              : the amount of consecutive frames to include in one input
    b_size           : batch size for training
    weight_DD        : weight to put on the error in discriminating real data
    traintotal       : total amount of files for training
    testtotal        : total amount of files for testing
    num_epochs       : number of epochs to run
    
    update_D_every   : for this many updates made to parameters of D network, we make one update to parameters of G network
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

    traintotal = traintotal
    print('Train file amount: {}'.format(traintotal))
    testtotal = testtotal
    print('Test file amount: {}'.format(testtotal))
    trainfiles = ncfiles[0:traintotal]
    testfiles = ncfiles[traintotal:traintotal+testtotal]
    print('weight of DD = ', weight_DD)
    assert((weight_DD>=0) and (weight_DD<=1))
    # Training Loop
    # Lists to keep track of progress
#     img_list = []
    G_losses   = []
    D_losses   = []
    val_losses = []
    criterion  = nn.BCELoss() if sigmoid_on else nn.BCEWithLogitsLoss()
    
    optimizerD = optim.Adam(netD.parameters(), lr=lrd, betas=(beta1, 0.999))
    schedulerD = ReduceLROnPlateau(optimizerD, 'min',factor=0.5,patience=3)
    optimizerG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, 0.999))
    schedulerG = StepLR(optimizerG, step_size=10, gamma=0.8)
    print("Starting Training Loop...")
    # For each epoch
#     with torch.autograd.set_detect_anomaly(True):

    fileexp  = ncfiles[fileexp_ind] # fix one file to observe the outcome of the generator
    simexp   = xr.open_dataarray(datapath + fileexp)
    dynexp   = np.zeros((1,1,dep,256,256))
    noiseexp = torch.zeros((1,1,dep,256,256)) # fix one set of dynamical frames for viewing
    for t in range(dep):
        dynexp[0,0,t,:,:] = resize(simexp.isel(t=t)[:img_size,:img_size].values,(256,256),anti_aliasing=True)
    normalize_factor   = np.max( dynexp[0,0,:,:,:].flatten() )
    dynexp[0,0,:,:,:]  = dynexp[0,0,:,:,:] / normalize_factor
    for t in range(dep): # different noise for each frame when using a 'for' loop
        noiseexp[0,0,t,:,:] = noise_generate(dynexp[0,0,t,:,:],mode='linear') 
    dynexp   = torch.tensor(dynexp).to(torch.float).to(device)
    
    for epoch in range(num_epochs):
        try:
            fileind = 0; global_step = 0; D_update_ind = 0
            while fileind < traintotal:
                dyn = np.zeros((b_size,1,dep,256,256))
                noise = torch.zeros(dyn.shape)
                bfile = 0
                while bfile < b_size:
                # Format batch: prepare training data for D network
                    filename = ncfiles[fileind+bfile]
                    sim = xr.open_dataarray(datapath+filename)
                    for t in range(dep):
                        dyn[bfile,0,t,:,:] = resize(sim.isel(t=t)[:img_size,:img_size].values,(256,256),anti_aliasing=True)
                    normalize_factor   = np.max( dyn[bfile,0,:,:,:].flatten() ) # normalize each File
                    dyn[bfile,0,:,:,:] = dyn[bfile,0,:,:,:] / normalize_factor
                    for t in range(dep): # different noise for each frame when using a 'for' loop
                        noise[bfile,0,t,:,:] = noise_generate(dyn[bfile,0,t,:,:],mode='linear')
                    sim.close()
                    bfile += 1
                dyn = torch.tensor(dyn).to(torch.float)
                real_cpu = dyn.to(device)
                fileind += b_size
    #             b_size = real_cpu.size(0)

                # Generate fake image batch with G
                fake = netG(noise + real_cpu)
            
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ############################
                if D_update_ind%update_D_every == 0:
                    ## Train with all-real batch           
                    netD.zero_grad()
                    Dx1_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    # Forward pass real batch through D
                    Dx1 = netD(real_cpu).view(-1)
                    # Calculate loss on all-real batch
                    errD_real = criterion(Dx1, Dx1_label)
                    # Calculate gradients for D in backward pass
                    errD_real.backward()
                    D_x = Dx1.mean().item() if sigmoid_on else torch.sigmoid(Dx1).mean().item()

                    ## Train with all-fake batch
                    DGz1_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        #                 label.fill_(fake_label)
                    # Classify all fake batch with D
                    DGz1 = netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterion(DGz1, DGz1_label)
                    # Calculate the gradients for this batch
                    errD_fake.backward(retain_graph=True)
                    D_G_z1 = DGz1.mean().item() if sigmoid_on else torch.sigmoid(DGz1).mean().item()
                    # Add the gradients from the all-real and all-fake batches
                    errD = weight_DD*errD_real + errD_fake
                    # Update D
                    optimizerD.step()           
                    D_losses.append(errD.item())    
                D_update_ind += 1

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ############################
                DGz2_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
    #                 label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                DGz2 = netD(fake).view(-1)
                D_G_z2 = DGz2.mean().item() if sigmoid_on else torch.sigmoid(DGz2).mean().item()
                # Calculate G's loss based on this output
                errG = criterion(DGz2, DGz2_label)
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
                if (global_step%print_every==0) and (len(D_losses)>0) :
                    print('[%d/%d][%d/%d]\t Loss_D: %.4f\t Loss_G: %.4f\t D(x): %.4f\t D(G(z)): D %.4f/0   G %.4f/1'
                          % (epoch+1, num_epochs, fileind, traintotal, \
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                if global_step%validate_every==0 :
                    val_loss = validate(testfiles,netD,netG,dep=dep,batchsize=5,\
                                        seed=manual_seed,img_size=img_size,sigmoid_on=sigmoid_on,device=device)
                    val_losses.append(val_loss.item())
                    print('validation loss = {}'.format(val_loss))
                    schedulerD.step(val_loss)
                    schedulerG.step()
                    if make_plot:
                        fakeexp = netG(noiseexp + dynexp)
                        illustrate(fakeexp) # should show a fixed set of images instead of the set under processing!
                        visualization(G_losses,D_losses,val_losses)                
                    if save_cp:
                        dir_checkpoint = '/home/huangz78/checkpoints/'
                        try:
                            os.mkdir(dir_checkpoint)
                            print('Created checkpoint directory')
                #                 logging.info('Created checkpoint directory')
                        except OSError:
                            pass
                        torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + 'netD.pth')
                        torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + 'netG.pth')
                        np.savez('/home/huangz78/checkpoints/gan_train_track.npz',\
                                 g_loss=G_losses,d_loss=D_losses,val_loss=val_losses)
                        print(f'\t Checkpoint saved at epoch {epoch + 1}, iteration {global_step}!')
                global_step += 1
        except KeyboardInterrupt: # need debug
            print('Keyboard Interrupted! Exit~')
            if save_cp:
                dir_checkpoint = '/home/huangz78/checkpoints/'
                torch.save({'model_state_dict': netD.state_dict()}, dir_checkpoint + 'netD.pth')
                torch.save({'model_state_dict': netG.state_dict()}, dir_checkpoint + 'netG.pth')
                np.savez('/home/huangz78/checkpoints/gan_train_track.npz',g_loss=G_losses,d_loss=D_losses,val_loss=val_losses)
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

