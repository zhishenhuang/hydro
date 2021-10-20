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



class wgan_trainer:
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
    def __init__(self,
                 netG: nn.Module,
                 netD: nn.Module,                 
                 dep:int=8,                               
                 img_size:int=320,
                 manual_seed:int=999,
                 resize_option:bool=False,
                 weight_super_decay:float=.97,
                 noise_mode:str='Abel-gaussian',
                 sigma:float=2,
                 volatility:float=.05,
                 xi:float=.02,
                 scaling:float=1.,
                 white_noise_ratio:float=1e-4,
                 normalize_factor:float=50.,
                 ngpu:int=0,
                 datapath:str='/mnt/DataB/hydro_simulations/data/',
                 dir_checkpoint:str = '/mnt/DataA/checkpoints/leo/hydro/',
                 dir_hist=None):             
        self.ngpu = ngpu
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else 'cpu')
        
        self.netG = netG.to(self.device)
        self.netD = netD.to(self.device)
        
        self.dep = dep        
        self.img_size = img_size
        self.manual_seed = manual_seed
        self.resize_option = resize_option
        self.noise_mode = noise_mode
        
        self.trainfiles = []
        self.valfiles   = []
        self.testfiles  = []
        self.normalize_factor = normalize_factor
        self.datapath = datapath
        self.dir_checkpoint = dir_checkpoint
        
        ## params for training
        self.weight_super_decay = weight_super_decay
        
        ## params for noise generation
        self.volatility = volatility
        self.sigma = sigma
        self.xi = xi
        self.scaling = scaling
        self.white_noise_ratio = white_noise_ratio
        
        self.dir_hist = dir_hist
        if self.dir_hist is None:
            self.G_losses  = []; self.D_losses = []; self.nrmse_train = list([]); self.l1_train = list([])
            self.Massdiffs = []; self.nrmse_val = []; self.l1_val = []
            self.bce_fake = [];  self.bce_real = []
            print('New training ~')
        else:
            histRec = np.load(self.dir_hist)
            self.G_losses = list(histRec[g_loss]); self.D_losses = list(histRec[d_loss])
            self.nrmse_train = list(histRec[nrmse_train]); self.l1_train = list(histRec[l1_train])
            self.Massdiffs = list(histRec[Massdiffs])
            self.nrmse_val = list(histRec[nrmse_val]); self.l1_val = list(histRec[l1_val])
            self.bce_fake = list(histRec[bce_fake]);  self.bce_real = list(histRec[bce_real])
            print('history of existing training record successfully loaded~' )     
        
    def empty_cache(self):
        torch.cuda.empty_cache()
        torch.backends.cuda.cufft_plan_cache.clear()
        
    def validate(self,batchsize):
        valfile_num = len(self.valfiles)
        batchsize = min(valfile_num,batchsize)
        # set the model in eval mode
        self.netD.eval()
        self.netG.eval()
        Mass_diff = 0; nrmse = 0; nl1err = 0
        # evaluate on validation set
        fileind = 0
        batch_step = 0
        with torch.no_grad():
            while fileind < valfile_num:
                dyn, noise = load_data_batch(fileind,self.valfiles,b_size=batchsize,dep=self.dep,img_size=self.img_size,\
                                            resize_option=self.resize_option,\
                                            noise_mode=self.noise_mode,normalize_factor = self.normalize_factor,\
                                            volatility=self.volatility,sigma=self.sigma,xi=self.xi,scaling=self.scaling,white_noise_ratio=self.white_noise_ratio)
                fileind += batchsize

                real_cpu = dyn.to(self.device)
                noise    = noise.to(self.device)
                fake = self.netG(noise + real_cpu).clamp(min=0).detach()
                fake[real_cpu==0] = 0

                mass_fake = compute_mass(fake,device=self.device)
                mass_real = compute_mass(real_cpu,device=self.device)
                mass_diff = torch.divide(torch.abs(mass_fake - mass_real), mass_real).sum()

                Mass_diff  = Mass_diff + mass_diff
                nrmse      = nrmse     + aver_mse(fake,real_cpu)  * fake.shape[0]
                nl1err     = nl1err    + aver_l1(fake,real_cpu)   * fake.shape[0]
                del dyn, real_cpu, noise 
        return Mass_diff/(valfile_num*self.dep), nrmse/valfile_num, nl1err/valfile_num
       
    def save_model(self,epoch=0,global_step=None):
        try:
            os.mkdir(self.dir_checkpoint)
            print('Created checkpoint directory')
        except OSError:
            pass
        torch.save({'model_state_dict': self.netD.state_dict()}, self.dir_checkpoint + f'netD_wg_{self.noise_mode}_scaling_{self.scaling}_supweigtdecay_{self.weight_super_decay}_epoch_{epoch}.pt')
        torch.save({'model_state_dict': self.netG.state_dict()}, self.dir_checkpoint + f'netG_wg_{self.noise_mode}_scaling_{self.scaling}_supweigtdecay_{self.weight_super_decay}_epoch_{epoch}.pt')
        np.savez(self.dir_checkpoint +f'wgan_train_track_{self.noise_mode}_scaling_{self.scaling}_supweigtdecay_{self.weight_super_decay}.npz',\
                         g_loss=self.G_losses,d_loss=self.D_losses,Massdiffs=self.Massdiffs,\
                         nrmse_train=self.nrmse_train,l1_train=self.l1_train,\
                         nrmse_val=self.nrmse_val,l1_val=self.l1_val,\
                         bce_real=self.bce_real,bce_fake=self.bce_fake)
        if isinstance(global_step,int):
            print(f'\t Checkpoint saved at epoch {epoch+1}, iteration {global_step+1}!')
        else:
            print(f'\t Checkpoint saved at epoch {epoch+1}!')
    
    def run(self,\
             lrd=1e-5, lrg=2e-5, b_size=5, b_size_test=5, beta1=0.9,\
             traintotal=500, testtotal=10, num_epochs=5,\
             weight_super=.99, weight_masscon=5, delta=0.2, weight_gradpen=5,\
             update_D_every=10, update_G_every=1,\
             save_cp=False, make_plot=False,print_every=10,
             epoch_start=0):
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)
        
        real_label = 1.
        fake_label = 0.
        use_cuda = True if (torch.cuda.is_available() and self.ngpu > 0) else False        
        ncfiles = list([])
        for file in os.listdir(self.datapath):
            if file.endswith(".nc"):
                ncfiles.append(file)
        print('Total amount of available files:', len(ncfiles))
        print('Train file amount: {}'.format(traintotal))
        print('Test file amount:  {}'.format(testtotal))
#         self.trainfiles = ncfiles[filestart:filestart+traintotal]
#         self.valfiles  = ncfiles[filestart+traintotal:filestart+traintotal+testtotal]         
        self.trainfiles = random.sample(set(ncfiles),k=traintotal)
        ncfiles = set(ncfiles) - set(self.trainfiles)
        self.valfiles  = random.sample(ncfiles,k=testtotal)
        ncfiles = set(ncfiles) - set(self.valfiles)
        self.testfiles = random.sample(ncfiles,k=200)
        np.savez(self.dir_checkpoint +f'filesUsed_{self.noise_mode}_scaling_{self.scaling}_supweigtdecay_{self.weight_super_decay}.npz',\
                         trainfiles=self.trainfiles,valfiles=self.valfiles,testfiles=self.testfiles)
        
        
        print('weight of mass conservation term in errG = ', weight_masscon)
        print('weight of supervised loss term in errG = ',   weight_super)
        assert(weight_masscon>=0)
        assert(weight_super>=0)
        
        # Training Loop
        # Lists to keep track of progress

        bce_loss   = nn.BCELoss()
        L1_loss    = nn.L1Loss()
        L2_loss    = nn.MSELoss()
        optimizerD = optim.Adam(self.netD.parameters(), lr=lrd, betas=(beta1, 0.999))
        schedulerD = StepLR(optimizerD,step_size=20,gamma=np.sqrt(.1))
#         schedulerD = ReduceLROnPlateau(optimizerD, 'min',factor=0.8,patience=20,min_lr=1e-7)
        optimizerG = optim.Adam(self.netG.parameters(), lr=lrg, betas=(beta1, 0.999))
        schedulerG = StepLR(optimizerG, step_size=40, gamma=0.9)
        print("Starting Training Loop...")
        
        global_step = 0;
        for epoch in range(epoch_start,num_epochs):
            try:
                fileind = 0; D_update_ind = 0; G_update_ind = 0
                bceR = 0; bceF = 0
                while fileind < traintotal:
                    # set the model back to training mode
                    self.netD.train()
                    self.netG.train()

                    dyn, noise = load_data_batch(fileind, self.trainfiles, \
                                                 b_size=b_size, dep=self.dep, img_size=self.img_size,\
                                                 resize_option=self.resize_option,\
                                                 noise_mode=self.noise_mode, normalize_factor=self.normalize_factor,\
                                                 volatility=self.volatility,sigma=self.sigma,xi=self.xi,scaling=self.scaling,white_noise_ratio=self.white_noise_ratio)
                    noise = noise.to(self.device)
                    real_cpu = dyn.to(self.device)
                    fileind += b_size

                    # Generate fake image batch with G
                    fake = self.netG(noise + real_cpu).clamp(min=0)
                    with torch.no_grad():
                        fake[real_cpu==0] = 0

                    l2err_tmp = aver_mse(fake,real_cpu); l1err_tmp = aver_l1(fake,real_cpu)
                    self.nrmse_train.append(l2err_tmp.item()); self.l1_train.append(l1err_tmp.item())
                    print(f"[{global_step+1}][{epoch+1}/{num_epochs}][{fileind}/{traintotal}]: l2err = {l2err_tmp.item()},  l1err = {l1err_tmp.item()}")
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ############################
                    if D_update_ind%update_D_every == 0:
                        ## Train with all-real batch           
                        optimizerD.zero_grad()
                        # Forward pass real batch through D
                        D_real_1 = self.netD(real_cpu).view(-1)
                        D_fake_1 = self.netD(fake.detach()).view(-1)
                        gradient_penalty = _gradient_penalty(self.netD,real_cpu,fake,use_cuda=use_cuda,\
                                                             gp_weight=weight_gradpen)

                        d_loss = D_fake_1.mean() - D_real_1.mean() + gradient_penalty
#                         print(f'D_fake_mean = {D_fake_1.mean().item()}, D_real_mean = {D_real_1.mean().item()}, grad pen. = {gradient_penalty}')
                        d_loss.backward()

                        optimizerD.step()           
                        self.D_losses.append(d_loss.item())
                        with torch.no_grad():
                            bce_loss_real = bce_loss(D_real_1,torch.ones_like(D_real_1)).detach()
                            bceR += bce_loss_real
                            self.bce_real.append(bce_loss_real.item())
                    D_update_ind += 1

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ############################
                    if G_update_ind%update_G_every == 0:

                        D_real_2 = self.netD(real_cpu).view(-1)
                        D_fake_2 = self.netD(fake).view(-1)
                        gradient_penalty = _gradient_penalty(self.netD,real_cpu,fake,use_cuda=use_cuda,\
                                                             gp_weight=weight_gradpen)

                        d_loss = D_fake_2.mean() - D_real_2.mean() + gradient_penalty

                        mass_fake = compute_mass(fake,device=self.device)
                        mass_fake.retain_grad()
                        mass_real = compute_mass(real_cpu,device=self.device)
                        g_loss = -(1-weight_super)*d_loss + weight_super*L2(fake,real_cpu) + delta*weight_super*L1(fake,real_cpu) + weight_masscon*L2_loss(mass_fake,mass_real)

                        # Calculate gradients for G
                        optimizerG.zero_grad()
                        g_loss.backward()
                        # Update G
                        optimizerG.step()
                        # Save Losses for plotting later
                        self.G_losses.append(g_loss.item())
                        with torch.no_grad():
                            bce_loss_fake = bce_loss(D_fake_2,torch.zeros_like(D_fake_2)).detach()
                            bceF += bce_loss_fake
                            self.bce_fake.append(bce_loss_fake.item())
                    G_update_ind += 1
                    del dyn, real_cpu, noise
                    self.empty_cache()
                    ############################
                    # Validation
                    ############################
                    if self.dir_hist is None:
                         val_condition = (fileind > traintotal-b_size) or ((epoch==0) and (global_step==0))
                    else:
                         val_condition = (fileind > traintotal-b_size)
                    if val_condition: # (b_size>abs(traintotal//2-fileind)) or  
                        massdiff,nrmse,l1err = self.validate(batchsize=b_size_test)
                        self.Massdiffs.append(massdiff.item())
                        self.nrmse_val.append(nrmse.item())
                        self.l1_val.append(l1err.item())
                        print(f'global step: {global_step+1} validation result \n mass difference = {massdiff.item()}, average nrmse = {nrmse.item()}, average l1 err = {l1err.item()}')
#                         schedulerD.step(bceR+bceF)
                        schedulerD.step()
                        bceR = 0; bceF = 0
                        schedulerG.step()
                    ############################
                    # Output training stats, and visualization
                    ############################
                    if (global_step%print_every==0) and (len(self.D_losses)>0) :
                        print(f'[{global_step+1}][{epoch+1}/{num_epochs}][{fileind}/{traintotal}]\t Loss_D: {d_loss.item()}\t Loss_G: {g_loss.item()}\t D(x): {D_real_2.mean().item()}\t D(G(z)): D {D_fake_1.mean().item()}/0   G {D_fake_2.mean().item()}/1')
                        if save_cp:
                            self.save_model(epoch=epoch,global_step=global_step)
                    global_step += 1
                weight_super *= self.weight_super_decay
                if save_cp:
                    self.save_model(epoch=epoch)
            except KeyboardInterrupt: # need debug
                print('Keyboard Interrupted! Exit~')
                if save_cp:
                    self.save_model(epoch=epoch,global_step=global_step)
                    print('D net, G net, and loss records are saved after key interrupt~')
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)

    
    
    
    
    
    
    
    
    
   
    

