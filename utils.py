import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
import torch
from scipy.ndimage import gaussian_filter
import xarray as xr
import random
import abel
# import importlib
# import logging
import os
import shutil
import sys
import copy
import torch.optim as optim
import h5py

plt.ioff()
plt.switch_backend('agg')

datapath = '/mnt/DataB/hydro_simulations/data/'

noisepath = '/mnt/DataB/hydro_simulations/noise_source.npz'
noisedata = np.load(noisepath)['data']

def lpnorm(x,xstar,p='fro',mode='sum'):
    '''
    x and xstar are both assumed to be in the format NCDHW
    '''
    assert(x.shape==xstar.shape)
    numerator   = torch.norm(x-xstar,p=p,dim=(3,4))
    denominator = torch.norm(xstar  ,p=p,dim=(3,4))
    error = torch.sum( torch.div(numerator,denominator) )
    if mode == 'sum':
        error = torch.sum(  torch.div(numerator,denominator) )
    elif mode == 'mean':
        error = torch.mean( torch.div(numerator,denominator) )
    
    return error

def L2(x,xstar,mode='sum'):
    return lpnorm(x,xstar,p='fro',mode=mode)
#     return torch.norm(x-xstar,'fro')/torch.norm(xstar,'fro')

def L1(x,xstar,mode='sum'):
    return lpnorm(x,xstar,p=1,mode=mode)
#     return torch.norm(x-xstar,p=1)/torch.norm(xstar,p=1)

def exp_negative( input , xi=1):
    '''
    assume the input is a image of the format HW
    '''
    return np.exp(-xi*input)

def exp_negative_inverse( input , xi=1):
    '''
    assume the input is a image of the format HW
    '''
    tmp = -1/xi * np.log(input)
    
    tmp[np.isnan(tmp)] = 0
    tmp[np.isinf(tmp)] = 0
    
    return tmp


from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def noise_generate(frame, \
                   fl=0.006, fr=0.0179, f=0.01, c=0.1, cl=0.06,cr=0.21, \
                   sigma=2,scaling=1,\
                   volatility=.04, xi=1,\
                   clampval=12, white_noise_ratio=.1,\
                   mode='linear',noisedata=noisedata,abel_method='basex'):
    if mode == 'linear':
        factor = np.random.uniform(fl,fr,1) * scaling
        const  = np.random.uniform(cl,cr,1) * scaling
        noise  = frame * factor + const
    elif mode == 'const':
        noise = (frame * f + c) * scaling
    elif mode == 'const_rand':
        c     = np.random.uniform(cl,cr,1)
        noise = torch.ones_like(frame) * c * scaling
    elif mode == 'gaussian':
        if len(frame.shape) == 2:
            noise_mag = white_noise_ratio*frame.abs().max()
            noise     = noise_mag*torch.randn(frame.shape[0],frame.shape[1])
        elif len(frame.shape) == 3: # assuming that there is only one channel
            noise = np.zeros_like(frame)
            for frame_ind in range(frame.shape[0]):
                noise_mag            = white_noise_ratio*frame[frame_ind,:,:].abs().max()
                noise[frame_ind,:,:] = noise_mag*torch.randn(frame.shape[1],frame.shape[2])
    elif mode == 'sampling':
        assert(clampval>0)
        maxheg = noisedata.shape[1]
        maxwid = noisedata.shape[2]
        if len(frame.shape) == 3:
            noise = torch.zeros_like(frame)
            heg,wid = frame.shape[1],frame.shape[2]
            for frame_ind in range(frame.shape[0]):
                noiseind  = np.random.randint(low=20, size=1, dtype=int)   
                noise_tmp = torch.tensor(noisedata[noiseind, maxheg-heg:, maxwid-wid:]).clamp(min=-clampval,max=clampval)
                scaling_rand = np.random.rand()
                noise[frame_ind,:,:] = \
                    noise_tmp/noise_tmp.abs().max() * frame[frame_ind,:,:].abs().max() * scaling_rand
        elif len(frame.shape) == 2:
            heg,wid = frame.shape[0],frame.shape[1]
            noiseind     = np.random.randint(low=20, size=1, dtype=int)
            noise_tmp    = torch.tensor(noisedata[noiseind, maxheg-heg:, maxwid-wid:]).clamp(min=-clampval,max=clampval)
            scaling_rand = np.random.rand()
            noise        = noise_tmp/noise_tmp.abs().max() * frame.abs().max() * scaling_rand
    elif mode == 'Abel-linear':        
        dyn_fullspan = torch_complete(frame)
        noise = torch.zeros_like(frame)
        assert(len(frame.shape)==3)
        heg = frame.shape[1]
        wid = frame.shape[2]
        ind = 0
        scaling_factor = np.random.rand()
        for img in dyn_fullspan:
            img_abel = abel.Transform(img.numpy(), method=abel_method, direction='forward',verbose=False).transform
            scatter_abel = gaussian_filter(img_abel,sigma,order=0)
            eps = np.random.uniform(low=-0.05,high=0.05)
            img_abel_noisy = img_abel +  ( (1+eps)*  scaling_factor ) * scatter_abel
#             img_abel_noisy = (1 + np.random.uniform(low=-volatility,high=volatility) ) * img_abel + \
#                     np.random.uniform(low=-volatility,high=volatility) * np.mean(np.abs(img_abel.flatten()))
#             img_abel_noisy = (1 + .04 ) * img_abel + .04 * np.mean(np.abs(img_abel.flatten()))
            img_noisy = abel.Transform(img_abel_noisy, method=abel_method, direction='backward',verbose=False).transform
            noise_fullspan = img_noisy - img.numpy()
            noise[ind,:,:] = torch.tensor(noise_fullspan[heg:,wid:])
            ind += 1
    elif mode == 'Abel-gaussian':        
        dyn_fullspan = torch_complete(frame)
        noise = torch.zeros_like(frame)
        assert(len(frame.shape)==3)
        heg = frame.shape[1]
        wid = frame.shape[2]
        ind = 0
        scaling_factor = scaling # np.random.rand()
        for img in dyn_fullspan:
            with suppress_stdout():
                img_abel = abel.Transform(img.numpy(), method=abel_method, direction='forward',verbose=False).transform
            D = exp_negative(img_abel,xi=xi)
            S = gaussian_filter(D,sigma,order=0)
            eps = np.random.uniform(low=-volatility,high=volatility)
            T = D + ( (1+eps)*  scaling_factor ) * S
            img_abel_noisy = exp_negative_inverse(T,xi=xi)
            with suppress_stdout():
                img_noisy = abel.Transform(img_abel_noisy, method=abel_method, direction='backward',verbose=False).transform
            noise_fullspan = img_noisy - img.numpy()
            noise[ind,:,:] = torch.tensor(noise_fullspan[heg:,wid:])
            ind += 1
    elif mode == 'Abel-gaussian-double':        
        dyn_fullspan = torch_complete(frame)
        noise = torch.zeros_like(frame)
        assert(len(frame.shape)==3)
        heg = frame.shape[1]
        wid = frame.shape[2]
        ind = 0
        scaling_factor = scaling # np.random.rand()
        for img in dyn_fullspan:
            with suppress_stdout():
                img_abel = abel.Transform(img.numpy(), method=abel_method, direction='forward',verbose=False).transform
            D = exp_negative(img_abel,xi=xi)
            S = gaussian_filter(D,sigma,order=0)
            eps = np.random.uniform(low=-volatility,high=volatility)
            T = D + ( (1+eps)*  scaling_factor ) * S
            img_abel_noisy = exp_negative_inverse(T,xi=xi)
            white_noise_magnitude = np.mean(img_abel_noisy) * white_noise_ratio
            img_abel_noisy = img_abel_noisy + white_noise_magnitude * np.random.randn(*(img_abel_noisy.shape))
            with suppress_stdout():
                img_noisy = abel.Transform(img_abel_noisy, method=abel_method, direction='backward',verbose=False).transform
            noise_fullspan = img_noisy - img.numpy()
            noise[ind,:,:] = torch.tensor(noise_fullspan[heg:,wid:])
            ind += 1
    return noise

def complete(a,sizefix=False):
    heg = a.shape[0]
    wid = a.shape[1]
    ar = np.concatenate((np.flipud(a),a), axis=0) # Flipping array to get the right part
    al = np.concatenate((np.flipud(np.fliplr(a)),np.fliplr(a)), axis=0) # Flipping array to get the left part
    # Combining to form a full circle from a quarter image and resizing the 700x700 to 200x200 image
    if sizefix:
        a_full = resize(np.concatenate((al,ar), axis=1), (heg, wid), anti_aliasing=True)
    else:
        a_full = np.concatenate((al,ar), axis=1)
    return a_full

def compute_mass(imgs,Rrho=1,Rz=1,device=torch.device('cpu')):
    '''
    computing through cylindrical coordinate
    input  --- imgs: [#batch, #channel, #dep, #heg, #wid]
    output --- mass: [#batch, #dep]
    '''
    drho = Rrho / imgs.shape[3]
    dz   = Rz   / imgs.shape[4]
    metrics = torch.linspace(0,Rrho,imgs.shape[4],device=device).repeat(imgs.shape[3],1)
    integrand = imgs * metrics
    mass = 2*np.pi * torch.sum(integrand,dim=(3,4)) * drho * dz 
    return torch.squeeze(mass)

def torch_complete(imgs):

    if len(imgs.shape)==5:   # assume the input img has the format NCDHW
        ar = torch.cat((torch.flip(imgs,dims=[3]),imgs), dim=3) # Flipping array to get the right part
        leftupquad = torch.flip(imgs,dims=[4])
        al = torch.cat((torch.flip(leftupquad,dims=[3]),leftupquad), dim=3) # Flipping array to get the left part
        # Combining to form a full circle from a quarter image
        a_full = torch.cat((al,ar), dim=(4))
    elif len(imgs.shape)==3: # assume the input img has the format DHW
        ar = torch.cat((torch.flip(imgs,dims=[1]),imgs), dim=1) # Flipping array to get the right part
        leftupquad = torch.flip(imgs,dims=[2])
        al = torch.cat((torch.flip(leftupquad,dims=[1]),leftupquad), dim=1) # Flipping array to get the left part
        # Combining to form a full circle from a quarter image
        a_full = torch.cat((al,ar), dim=(2))
    else:
        raise('Input dimension for the torch_complete function is incorrect.')
    return a_full

def load_data_batch(fileind,trainfiles,b_size=5,dep=8,img_size=320,\
                    resize_option=False,newsize=256,\
                    noise_mode='const_rand',normalize_factor=50,\
                    sigma=2,volatility=.04,xi=.02,scaling=1.,white_noise_ratio=.1):
    traintotal = len(trainfiles)
    assert(dep<41)
    time_pts = torch.round(torch.linspace(0,40,dep)).int() 
    current_b_size = min(b_size,traintotal-fileind)
    if not resize_option:
        dyn = torch.zeros((current_b_size,1,dep,img_size,img_size))
    else:
        dyn = torch.zeros((current_b_size,1,dep,newsize,newsize))
    noise = torch.zeros(dyn.shape)
    bfile = 0
    while (bfile < current_b_size) and (fileind+bfile < traintotal):
    # Format batch: prepare training data for D network
        filename = trainfiles[fileind+bfile]
        sim = xr.open_dataarray(datapath+filename)
        for t in range(dep):
            if not resize_option:
                dyn[bfile,0,t,:,:] = torch.tensor(sim.isel(t=time_pts[t])[:img_size,:img_size].values, dtype=torch.float).clamp(max=normalize_factor)
            else:
                dyn[bfile,0,t,:,:] = torch.tensor(
                    resize(sim.isel(t=time_pts[t])[:img_size,:img_size].values,(newsize,newsize),anti_aliasing=True), dtype=torch.float).clamp(max=normalize_factor)

#         maxval_tmp = torch.max( torch.abs(dyn[bfile,0,:,:,:]) ) # normalize each File
#         if maxval_tmp > normalize_factor:
#             bfile -= 1
#             fileind += 1
#         else:                        
        dyn[bfile,0,:,:,:]   = dyn[bfile,0,:,:,:] / normalize_factor
        noise[bfile,0,:,:,:] = noise_generate(dyn[bfile,0,:,:,:],mode=noise_mode,\
                                              scaling=scaling,\
                                              sigma=sigma,volatility=volatility,xi=xi,white_noise_ratio=white_noise_ratio) # scaling = dyn[bfile,0,:,:,:].max()
        sim.close()
        bfile += 1
    return dyn, noise

def illustrate(imgs,\
               vmin=0,vmax=1.1,\
               title=None,\
               fontsize=12,figsize=(18,9),cbar_shrink=.9,dpi=150,\
               time_pts=None,\
               save_path=None,\
               title_on=True,\
               nrows=2,ncols=4,\
               dep=8,\
               complete_on=False):
    samp = imgs.clone().detach()
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, figsize=figsize)
    if (vmin==-np.inf) or (vmax==np.inf):
        if complete_on:
            vmax = torch.max(complete(samp[0,0,:,:,:]))
            vmin = torch.min(complete(samp[0,0,:,:,:]))
        else:
            vmax = torch.max(samp[0,0,:,:,:])
            vmin = torch.min(samp[0,0,:,:,:])
    assert(dep<=nrows*ncols)
    for t in range(dep):
        if time_pts is not None:
            axs[t//ncols][t%ncols].set_title('t = ' + str(time_pts[t].item()))
        else:
            axs[t//ncols][t%ncols].set_title('t = ' + str(t))
        if complete_on:
            hd = axs[t//ncols][t%ncols].imshow(complete(samp[0,0,t,:,:]),vmin=vmin, vmax=vmax,origin='lower')
        else:
            hd = axs[t//ncols][t%ncols].imshow(samp[0,0,t,:,:],vmin=vmin, vmax=vmax,origin='lower')
#         divider = make_axes_locatable(axs[t//4][t%4])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         fig.colorbar(hd,cax=cax)
    cbar = fig.colorbar(hd, ax=axs.ravel().tolist(), shrink=cbar_shrink)
    cbar.set_ticks(np.arange(vmin, vmax, (vmax-vmin)/10))
    if title_on:
        if title is None:
            fig.suptitle('Generated dynamics, ' + str(t) + ' consecutive frames')
        else:
            fig.suptitle(title)
    plt.rcParams.update({'font.size': fontsize})
#     plt.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path,format='eps',dpi=dpi,transparent=True,bbox_inches='tight')
    print('\n')

def rolling_mean(x,window):
    window = int(window)
#   y = np.zeros(x.size-window)
#   for ind in range(y.size):
#       y[ind] = np.mean(x[ind:ind+window])

    # Stephen: for large data, the above gets a bit slow, so we can do this:
#   y = np.convolve(x, np.ones(window)/window, mode='valid')
#   return y
    # or https://stackoverflow.com/a/27681394
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

# def visualization(G_losses,D_losses,val_losses,nrmse_=None,l1err_=None,linferr_=None,\
#                   log_loss=False,log_val=False,log_err=False,window=0):
#     fig,axs = plt.subplots(3,1,figsize=(10,8))
#     axs[0].set_xlabel('iters', fontsize=16)
#     axs[0].set_ylabel('G loss', color='r', fontsize=16)
#     if window > 0:
#         axs[0].plot(rolling_mean(G_losses,window), color='r')
#     else:
#         axs[0].plot(G_losses, color='r')
#     axs[0].tick_params(axis='x', labelsize='large')
#     axs[0].tick_params(axis='y', labelcolor='r',labelsize='large')
#     if log_loss:
#         axs[0].set_yscale('log')
    
#     axs[1].set_xlabel('iters', fontsize=16)
#     axs[1].set_ylabel('D loss', color='b', fontsize=16)
#     if window > 0:
#         axs[1].plot(rolling_mean(D_losses,window), color='b')
#     else:
#         axs[1].plot(D_losses, color='b')
#     axs[1].tick_params(axis='y', labelcolor='b',labelsize='large')
#     if log_loss:
#         axs[1].set_yscale('log')

#     axs[2].set_xlabel('iters', fontsize=16)
#     axs[2].set_ylabel('validation loss', fontsize=16)
#     axs[2].plot(val_losses)
#     if log_val:
#         axs[2].set_yscale('log')
#     axs[2].tick_params(axis='x', labelsize='large')
#     axs[2].tick_params(axis='y', labelsize='large')
#     plt.show()
    
#     if nrmse_ is not None:
#         plt.figure()        
#         plt.xlabel('iters', fontsize=16)
#         plt.ylabel('nrmse, averaged', fontsize=16)
#         plt.plot(nrmse_)
#         if log_err:
#             plt.yscale('log')
#         plt.tick_params(axis='x', labelsize='large')
#         plt.tick_params(axis='y', labelsize='large')
#         plt.show()
#     if l1err_ is not None:
#         plt.figure()       
#         plt.xlabel('iters', fontsize=16)
#         plt.ylabel('rel. l1 error, averaged', fontsize=16)
#         plt.plot(l1err_)
#         if log_err:
#             plt.yscale('log')
#         plt.tick_params(axis='x', labelsize='large')
#         plt.tick_params(axis='y', labelsize='large')
#         plt.show()
        
#     if linferr_ is not None:
#         plt.figure()        
#         plt.xlabel('iters', fontsize=16)
#         plt.ylabel('rel. linf error, averaged', fontsize=16)
#         plt.plot(linferr_)
#         if log_err:
#             plt.yscale('log')
#         plt.tick_params(axis='x', labelsize='large')
#         plt.tick_params(axis='y', labelsize='large')   
#         plt.show()

def visualization(data,labels,\
                  log=False,window=0,figsize=(15,25),fontsize=16):
    nums = len(data)
    colors = ['r','b','g','tab:orange','tab:purple','tab:brown','k',\
              'deeppink','darkblue','darkorange','darkslategray','darkseagreen']
    fig,axs = plt.subplots(nums,1,figsize=figsize)
    for ind in range(nums):
        axs[ind].set_xlabel('iters', fontsize=fontsize)
        axs[ind].set_ylabel(labels[ind], color=colors[ind], fontsize=fontsize)
        if window > 0:
            axs[ind].plot(rolling_mean(data[labels[ind]],window), color=colors[ind])
        else:
            axs[ind].plot(data[labels[ind]], color=colors[ind])
        axs[ind].tick_params(axis='x', labelsize='large')
        axs[ind].tick_params(axis='y', labelcolor=colors[ind],labelsize='large')
        if log:
            axs[ind].set_yscale('log')
    plt.show()
    
def TVA(X):
    '''
    X is in the format NCDHW
    '''
    assert(len(X.shape)==5)
    diff1 = X[:,:,:,1:,:] - X[:,:,:,0:-1,:]
    diff2 = X[:,:,:,:,1:] - X[:,:,:,:,0:-1]
    tva   = torch.norm(diff1,p=1) + torch.norm(diff2,p=1)
    return tva

def postprocessor(noisy_dyn,truemass,\
                  lr=1e-3,maxIter=2000,\
                  weight_datafid=1, weight_masscon=1e2,weight_TVA=1e-4,\
                  print_every=50,dyn=None,device=torch.device('cpu'),verbose=True):    
    # loss function = L_datafid * ||x - G(noisy_dyn)||_2^2 + L_massfid *|| compute_mass(x) - true_mass ||_2^2 + L_TVA * TVA(x)
    
    maxIter = int(maxIter)
    if dyn is None: # ground truth not available
        denoised_dyn_init = copy.deepcopy(noisy_dyn).to(device)
        dyn_new           = copy.deepcopy(noisy_dyn).to(device) + torch.randn(noisy_dyn.shape,device=device)*1e-5
    else: # ground truth dyn available
        denoised_dyn_init = copy.deepcopy(dyn).to(device)
        dyn_new           = copy.deepcopy(noisy_dyn).to(device)
    
    dyn_new.requires_grad = True  
    optimizer = optim.RMSprop( [{'params': dyn_new}],lr=lr,weight_decay=0 )
    for t in range(maxIter):
        mass_denoised = compute_mass(dyn_new,device=device)
        data_fidelity = lpnorm(dyn_new , denoised_dyn_init, p='fro', mode='mean')
        mass_fidelity = torch.norm(mass_denoised - truemass)  
        tva           = TVA(dyn_new)        
        loss =  weight_datafid * data_fidelity + weight_masscon * mass_fidelity + weight_TVA * tva        
        if (t%print_every==0) and verbose:
            print(f'iter {t:4d}, loss total {loss.data:5f}, data fid. {data_fidelity:5f}, mass fid. {weight_masscon * mass_fidelity:5f}, TVA {weight_TVA * tva:5f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if dyn is not None:
            with torch.no_grad():
                dyn_new[dyn==0] = 0

    return dyn_new.data
        
def aver_mse(fake,real):
    '''
    assume both input are 5-dimensional tensors
    '''
    diff  = torch.sqrt( torch.sum(torch.square(fake - real),dim=(2,3,4)) )
    denom = torch.sqrt( torch.sum(torch.square(real) , dim=(2,3,4)) )
    nrmse = torch.sum(torch.div(diff,denom),dim=(0))/fake.shape[0]
    return nrmse

def aver_l1(fake,real):
    '''
    assume both input are 5-dimensional tensors
    '''
    diff   = torch.sum( torch.abs(fake - real),dim=(2,3,4) ) 
    denom  = torch.sum( torch.abs(real), dim=(2,3,4) ) 
    nl1err = torch.sum(torch.div(diff,denom),dim=(0))/fake.shape[0]
    return nl1err

def aver_linf(fake,real):
    '''
    assume both input are 5-dimensional tensors
    '''
    diff     = torch.flatten(torch.abs(fake - real),start_dim=2)
    denom    = torch.flatten(torch.abs(real),start_dim=2)
    reldev   = torch.div(diff,denom)
    reldev[reldev==float("Inf")] = 0
    reldev   = torch.max(reldev,dim=2)[0]
    nlinferr = torch.sum(reldev,dim=(0))/fake.shape[0]
    return nlinferr


def test_(netG, testfiles,\
          batchsize=5,dep=8,img_size=320,\
          noise_mode='Abel-gaussian',\
          normalize_factor=50,\
          volatility=.05,sigma=2,xi=.02,scaling=1,white_noise_ratio=1e-4,device=torch.device('cpu'),\
          resize_option=False,postprocess=False,\
          maxIter=7e3,weight_datafid=0,weight_masscon=1e2,weight_TVA=1e-4):
    testfile_num = len(testfiles)
    batchsize = min(testfile_num,batchsize)
    # set the model in eval mode
    netG.to(device)
    netG.eval()
    Mass_diff = np.zeros(len(testfiles)); nrmse = np.zeros(len(testfiles)); nl1err = np.zeros(len(testfiles))

    # evaluate on validation set
    fileind = 0
    batch_step = 0    
    while fileind < testfile_num:   
        with torch.no_grad():
            dyn, noise = load_data_batch(fileind,testfiles,b_size=batchsize,dep=dep,img_size=img_size,\
                                        resize_option=resize_option,\
                                        noise_mode=noise_mode,normalize_factor = normalize_factor,\
                                        volatility=volatility,sigma=sigma,xi=xi,scaling=scaling,\
                                         white_noise_ratio=white_noise_ratio)
            real_cpu = dyn.to(device)
            noise    = noise.to(device)        
            fake = netG(noise + real_cpu).clamp(min=0).detach()
            fake[real_cpu==0] = 0

            mass_real = compute_mass(real_cpu,device=device)
        if postprocess:
            fake = postprocessor(fake,mass_real,\
                             lr=1e-5,\
                             weight_datafid=weight_datafid, weight_masscon=weight_masscon, weight_TVA=weight_TVA,\
                             dyn=dyn,\
                             maxIter=maxIter,\
                             print_every=500,device=device) # denoised_dyn

        mass_fake = compute_mass(fake,device=device)
        mass_diff = torch.divide(torch.abs(mass_fake - mass_real), mass_real).sum()/dep
        for ind in range(batchsize):
            Mass_diff[fileind+ind] = mass_diff
            nrmse[fileind+ind]     = aver_mse(fake[ind:ind+1,:,:,:,:],real_cpu[ind:ind+1,:,:,:,:])
            nl1err[fileind+ind]    = aver_l1(fake[ind:ind+1,:,:,:,:],real_cpu[ind:ind+1,:,:,:,:])
        del dyn, real_cpu, noise 
        fileind += batchsize
        print(f'[{fileind}/{testfile_num}]')
    return Mass_diff, nrmse, nl1err 

def test_baseline(testfiles,\
          batchsize=5,dep=8,img_size=320,\
          noise_mode='Abel-gaussian',\
          normalize_factor=50,\
          volatility=.05,sigma=2,xi=.02,scaling=1,white_noise_ratio=1e-4,device=torch.device('cpu'),\
          resize_option=False,\
          lr=1e-5,weight_datafid=0,weight_masscon=1e2,weight_TVA=1e-4,maxIter=5e3):
    testfile_num = len(testfiles)
    batchsize = min(testfile_num,batchsize)
    
    Mass_diff = np.zeros(len(testfiles)); nrmse = np.zeros(len(testfiles)); nl1err = np.zeros(len(testfiles))
    # evaluate on validation set
    fileind = 0
    batch_step = 0
    while fileind < testfile_num:
        print(f'Current iter: [{fileind}/{testfile_num}]')
        dyn, noise = load_data_batch(fileind,testfiles,b_size=batchsize,dep=dep,img_size=img_size,\
                                        resize_option=resize_option,\
                                        noise_mode=noise_mode,normalize_factor=normalize_factor,\
                                        volatility=volatility,sigma=sigma,xi=xi,scaling=scaling,\
                                        white_noise_ratio=white_noise_ratio)
        real_cpu = dyn.to(device)
        noise    = noise.to(device)
        noisy_sg = noise + real_cpu

        truemass = compute_mass(real_cpu,device=device)
        denoised_dyn = postprocessor(noisy_sg, truemass,\
                                     lr=lr,\
                                     weight_datafid=weight_datafid, weight_masscon=weight_masscon, weight_TVA=weight_TVA,\
                                     dyn=real_cpu,\
                                     maxIter=maxIter,\
                                     verbose=False,device=device)

        mass_fake = compute_mass(denoised_dyn,device=device)
        mass_real = compute_mass(real_cpu,device=device)
        mass_diff = torch.divide(torch.abs(mass_fake - mass_real), mass_real).sum()/dep
        for ind in range(batchsize):
            Mass_diff[fileind+ind] = mass_diff
            nrmse[fileind+ind]     = aver_mse(denoised_dyn[ind:ind+1,:,:,:,:],real_cpu[ind:ind+1,:,:,:,:])
            nl1err[fileind+ind]    = aver_l1(denoised_dyn[ind:ind+1,:,:,:,:],real_cpu[ind:ind+1,:,:,:,:])
        del dyn, real_cpu, noise 
        fileind += batchsize
            
    return Mass_diff, nrmse, nl1err
