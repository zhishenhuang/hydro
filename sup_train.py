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

class sup_trainer:
    '''
    netG             : input Generative network
    lrg              : learning rate for G network.
    dep              : the amount of consecutive frames to include in one input
    b_size           : batch size for training
    weight_masscon   : weight on the error in mass conservation
    delta            : weight on the relative L2 loss of data fidelity in the W-GAN G loss
    traintotal       : total amount of files for training
    testtotal        : total amount of files for testing
    num_epochs       : number of epochs to run
    
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
                 dep:int=8,                               
                 img_size:int=320,
                 manual_seed:int=999,
                 resize_option:bool=False,
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
                 dir_hist=None,
                 pure_gan=False):             
        self.ngpu = ngpu
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and (self.ngpu > 0)) else 'cpu')
        
        self.netG = netG.to(self.device)
        
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
        
        
        ## params for noise generation
        self.volatility = volatility
        self.sigma = sigma
        self.xi = xi
        self.scaling = scaling
        self.white_noise_ratio = white_noise_ratio
        
        self.dir_hist = dir_hist
        if self.dir_hist is None:
            self.G_losses  = []; self.nrmse_train = list([]); self.l1_train = list([])
            self.Massdiffs = []; self.nrmse_val = []; self.l1_val = []
            self.bce_fake = [];  self.bce_real = []
            print('New training ~')
        else:
            histRec = np.load(self.dir_hist)
            self.G_losses = list(histRec['g_loss'])
            self.nrmse_train = list(histRec['nrmse_train']); self.l1_train = list(histRec['l1_train'])
            self.Massdiffs = list(histRec['Massdiffs'])
            self.nrmse_val = list(histRec['nrmse_val']); self.l1_val = list(histRec['l1_val'])
            print('history of existing training record successfully loaded~' )   
        
    def empty_cache(self):
        torch.cuda.empty_cache()
        torch.backends.cuda.cufft_plan_cache.clear()
        
    def validate(self,batchsize):
        valfile_num = len(self.valfiles)
        batchsize   = min(valfile_num,batchsize)
        # set the model in eval mode
        self.netG.eval()
        Mass_diff = 0; nrmse = 0; nl1err = 0
        # evaluate on validation set
        fileind = 0
        with torch.no_grad():
            while fileind < valfile_num:
                dyn, noise = load_data_batch(fileind,self.valfiles,b_size=batchsize,dep=self.dep,img_size=self.img_size,\
                                            resize_option=self.resize_option,\
                                            noise_mode=self.noise_mode,normalize_factor = self.normalize_factor,\
                                            volatility=self.volatility,sigma=self.sigma,xi=self.xi,scaling=self.scaling,\
                                            white_noise_ratio=self.white_noise_ratio)
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
                del dyn, real_cpu, noise, fake
                self.empty_cache()
        
        return Mass_diff/(valfile_num*self.dep), nrmse/valfile_num, nl1err/valfile_num
       
    def save_model(self,epoch=0,global_step=None):
        try:
            os.mkdir(self.dir_checkpoint)
            print('Created checkpoint directory')
        except OSError:
            pass
        
        setting_name = f'sup_{self.noise_mode}_scaling_{self.scaling}_masswegt_{self.weight_masscon}'
        
        epoch_note = f'_epoch{epoch}_'
        
        torch.save({'model_state_dict': self.netG.state_dict()}, self.dir_checkpoint + 'netG_' + setting_name + epoch_note + '.pt')
        np.savez(self.dir_checkpoint + 'sup_train_track_' + setting_name + '.npz',\
                         g_loss=self.G_losses,Massdiffs=self.Massdiffs,\
                         nrmse_train=self.nrmse_train,l1_train=self.l1_train,\
                         nrmse_val=self.nrmse_val,l1_val=self.l1_val,)
        if isinstance(global_step,int):
            print(f'\t Checkpoint saved at epoch {epoch+1}, iteration {global_step+1}!')
        else:
            print(f'\t Checkpoint saved at epoch {epoch+1}!')
    
    def run(self,\
             lrg=2e-5, b_size=5, b_size_test=5, beta1=0.9,\
             traintotal=500, testtotal=10, num_epochs=5,\
             weight_masscon=5, delta=0.2,\
             save_cp=False, make_plot=False,print_every=10,
             epoch_start=0):
        random.seed(self.manual_seed)
        torch.manual_seed(self.manual_seed)
        
        self.weight_masscon = weight_masscon

        use_cuda = True if (torch.cuda.is_available() and self.ngpu > 0) else False        
        ncfiles = list([])
        for file in os.listdir(self.datapath):
            if file.endswith(".nc"):
                ncfiles.append(file)
        print('Total amount of available files:', len(ncfiles))
        print('Train file amount: {}'.format(traintotal))
        print('Test file amount:  {}'.format(testtotal))
        
        self.trainfiles = random.sample(set(ncfiles),k=traintotal)
        ncfiles = set(ncfiles) - set(self.trainfiles)
        self.valfiles  = random.sample(ncfiles,k=testtotal)
        ncfiles = set(ncfiles) - set(self.valfiles)
        self.testfiles = random.sample(ncfiles,k=200)
        np.savez(self.dir_checkpoint +f'sup_filesUsed_{self.noise_mode}_scaling_{self.scaling}_masswgt_{weight_masscon}.npz',\
                         trainfiles=self.trainfiles,valfiles=self.valfiles,testfiles=self.testfiles)
        
        
        print('weight of mass conservation term in errG = ', weight_masscon)
        
        assert(weight_masscon>=0)
        # Training Loop
        # Lists to keep track of progress

        bce_loss   = nn.BCELoss()
        L1_loss    = nn.L1Loss()
        L2_loss    = nn.MSELoss()
        
        optimizerG = optim.Adam(self.netG.parameters(), lr=lrg, betas=(beta1, 0.999))
        schedulerG = StepLR(optimizerG, step_size=40, gamma=0.95, verbose=True)
        print("Starting Training Loop...")
        
        global_step = 0;
        for epoch in range(epoch_start,num_epochs):
            try:
                fileind = 0
                while fileind < traintotal:
                    # set the model back to training mode
                    self.netG.train()                       
                    dyn, noise = load_data_batch(fileind, self.trainfiles, \
                                                 b_size=b_size, dep=self.dep, img_size=self.img_size,\
                                                 resize_option=self.resize_option,\
                                                 noise_mode=self.noise_mode, normalize_factor=self.normalize_factor,\
                                                 volatility=self.volatility,sigma=self.sigma,xi=self.xi,scaling=self.scaling,white_noise_ratio=self.white_noise_ratio)
                    noise    = noise.to(self.device)
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
                    # (2) Update G network: maximize log(D(G(z)))
                    ############################
                   
                    mass_fake = compute_mass(fake,device=self.device)
                    mass_fake.retain_grad()
                    mass_real = compute_mass(real_cpu,device=self.device)
                    g_loss = L2(fake,real_cpu) + weight_masscon*L2_loss(mass_fake,mass_real)

                    # Calculate gradients for G
                    optimizerG.zero_grad()
                    g_loss.backward()
                    # Update G
                    optimizerG.step()
                    # Save Losses for plotting later
                    self.G_losses.append(g_loss.item())
                    
                    del dyn, real_cpu, noise, fake
                    self.empty_cache()
                    
                    ############################
                    # Validation
                    ############################
                    if self.dir_hist is None:
                         val_condition = (fileind > traintotal-b_size) or ( (epoch==0) and (global_step==0) )
                    else:
                         val_condition = (fileind > traintotal-b_size)
                    if val_condition: # (b_size>abs(traintotal//2-fileind)) or  
                        massdiff,nrmse,l1err = self.validate(batchsize=b_size_test)
                        self.Massdiffs.append(massdiff.item())
                        self.nrmse_val.append(nrmse.item())
                        self.l1_val.append(l1err.item())
                        print(f'global step: {global_step+1} validation result \n mass difference = {massdiff.item()}, average nrmse = {nrmse.item()}, average l1 err = {l1err.item()}')
                        
                        schedulerG.step()
                    
                    ############################
                    # Output training stats, and visualization
                    ############################
                    if global_step%print_every==0:
                        print(f'[{global_step+1}][{epoch+1}/{num_epochs}][{fileind}/{traintotal}]\t Loss_G: {g_loss.item()}\t')
                        if save_cp:
                            self.save_model(epoch=epoch,global_step=global_step)
                    global_step += 1
                if save_cp:
                    self.save_model(epoch=epoch)
            except KeyboardInterrupt: # need debug
                print('Keyboard Interrupted! Exit~')
                if save_cp:
                    self.save_model(epoch=epoch,global_step=global_step)
                    print('G net, and loss records are saved after key interrupt~')
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)
