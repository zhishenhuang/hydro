import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
import torch

import xarray as xr
import random

import importlib
import logging
import os
import shutil
import sys

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

def noise_generate(frame, \
                   fl=0.006, fr=0.0179, f=0.01, c=0.1, cl=0.06,cr=0.21, sigma=1e-1, \
                   scaling=1,clampval=12,\
                   mode='linear',noisedata=noisedata):
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
            noise_mag = sigma*frame.abs().max()
            noise = noise_mag*torch.randn(frame.shape[0],frame.shape[1])
        elif len(frame.shape) == 3: # assuming that there is only one channel
            noise = np.zeros_like(frame)
            for frame_ind in range(frame.shape[0]):
                noise_mag            = sigma*frame[frame_ind,:,:].abs().max()
                noise[frame_ind,:,:] = noise_mag*torch.randn(frame.shape[1],frame.shape[2])
    elif mode == 'real':
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

def load_data_batch(fileind,trainfiles,b_size=5,dep=8,img_size=320,\
                    resize_option=False,newsize=256,\
                    noise_mode='const_rand',normalize_factor=50):
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
        noise[bfile,0,:,:,:] = noise_generate(dyn[bfile,0,:,:,:],mode=noise_mode,scaling=dyn[bfile,0,:,:,:].max())
        sim.close()
        bfile += 1
    return dyn, noise

def illustrate(imgs,full=False,vmin=0,vmax=1.1,title=None,fontsize=12,figsize=(18,9),cbar_shrink=.9,time_pts=None):
    samp = imgs.clone().detach()
#     vmax = torch.max(torch.flatten(samp))
#     vmin = torch.min(torch.flatten(samp))
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex=False, figsize=figsize)
    if vmin==-np.inf or vmax==np.inf:
        if full:
            vmax = torch.max(complete(samp[0,0,:,:,:]))
            vmin = torch.min(complete(samp[0,0,:,:,:]))
        else:
            vmax = torch.max(samp[0,0,:,:,:])
            vmin = torch.min(samp[0,0,:,:,:])
    for t in range(8):
        if time_pts is not None:
            axs[t//4][t%4].set_title('t = ' + str(time_pts[t].item()))
        else:
            axs[t//4][t%4].set_title('t = ' + str(t))
        if full:
            hd = axs[t//4][t%4].imshow(complete(samp[0,0,t,:,:]),vmin=vmin, vmax=vmax,origin='lower')
        else:
            hd = axs[t//4][t%4].imshow(samp[0,0,t,:,:],vmin=vmin, vmax=vmax,origin='lower')
#         divider = make_axes_locatable(axs[t//4][t%4])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         fig.colorbar(hd,cax=cax)
    cbar = fig.colorbar(hd, ax=axs.ravel().tolist(), shrink=cbar_shrink)
    cbar.set_ticks(np.arange(vmin, vmax, (vmax-vmin)/10))
    if title is None:
        fig.suptitle('Generated dynamics, ' + str(t) + ' consecutive frames')
    else:
        fig.suptitle(title)
    plt.rcParams.update({'font.size': fontsize})
#     plt.tight_layout()
    plt.show()
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

def visualization(G_losses,D_losses,val_losses,nrmse_=None,l1err_=None,linferr_=None,\
                  log_loss=False,log_val=False,log_err=False,window=0):
    fig,axs = plt.subplots(3,1,figsize=(10,8))
    axs[0].set_xlabel('iters', fontsize=16)
    axs[0].set_ylabel('G loss', color='r', fontsize=16)
    if window > 0:
        axs[0].plot(rolling_mean(G_losses,window), color='r')
    else:
        axs[0].plot(G_losses, color='r')
    axs[0].tick_params(axis='x', labelsize='large')
    axs[0].tick_params(axis='y', labelcolor='r',labelsize='large')
    if log_loss:
        axs[0].set_yscale('log')
    
    axs[1].set_xlabel('iters', fontsize=16)
    axs[1].set_ylabel('D loss', color='b', fontsize=16)
    if window > 0:
        axs[1].plot(rolling_mean(D_losses,window), color='b')
    else:
        axs[1].plot(D_losses, color='b')
    axs[1].tick_params(axis='y', labelcolor='b',labelsize='large')
    if log_loss:
        axs[1].set_yscale('log')

    axs[2].set_xlabel('iters', fontsize=16)
    axs[2].set_ylabel('validation loss', fontsize=16)
    axs[2].plot(val_losses)
    if log_val:
        axs[2].set_yscale('log')
    axs[2].tick_params(axis='x', labelsize='large')
    axs[2].tick_params(axis='y', labelsize='large')
    plt.show()
    
    if nrmse_ is not None:
        plt.figure()        
        plt.xlabel('iters', fontsize=16)
        plt.ylabel('nrmse, averaged', fontsize=16)
        plt.plot(nrmse_)
        if log_err:
            plt.yscale('log')
        plt.tick_params(axis='x', labelsize='large')
        plt.tick_params(axis='y', labelsize='large')
        plt.show()
    if l1err_ is not None:
        plt.figure()       
        plt.xlabel('iters', fontsize=16)
        plt.ylabel('rel. l1 error, averaged', fontsize=16)
        plt.plot(l1err_)
        if log_err:
            plt.yscale('log')
        plt.tick_params(axis='x', labelsize='large')
        plt.tick_params(axis='y', labelsize='large')
        plt.show()
        
    if linferr_ is not None:
        plt.figure()        
        plt.xlabel('iters', fontsize=16)
        plt.ylabel('rel. linf error, averaged', fontsize=16)
        plt.plot(linferr_)
        if log_err:
            plt.yscale('log')
        plt.tick_params(axis='x', labelsize='large')
        plt.tick_params(axis='y', labelsize='large')   
        plt.show()

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
