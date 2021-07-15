import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import random

def Nrmse(x,xstar):
    return torch.norm(x-xstar,'fro')/torch.norm(xstar,'fro')

def l1(x,xstar):
    return torch.norm(x-xstar,p=1)/torch.norm(xstar,p=1)

def noise_generate(frame, fl=0.006, fr=0.0179, cl=0.06,cr=0.21, sigma=1e-1, f=0.01, c=0.1 ,mode='linear' ):
    if mode == 'linear':
        factor = np.random.uniform(fl,fr,1)
        const  = np.random.uniform(cl,cr,1)
        noise  = frame * factor + const
    elif mode == 'const':
        noise = frame * f + c
    elif mode == 'const_rand':
        c     = np.random.uniform(cl,cr,1)
        noise = np.ones(frame.shape) * c
    elif mode == 'gaussian':
        noise_mag = sigma*np.max(np.abs(frame.flatten()))
        noise = noise_mag*np.random.randn(frame.shape[0],frame.shape[1])
    return noise

def complete(a):
    heg = a.shape[0]
    wid = a.shape[1]
    ar=np.concatenate((np.flipud(a),a), axis=0) # Flipping array to get the right part
    al=np.concatenate((np.flipud(np.fliplr(a)),np.fliplr(a)), axis=0) # Flipping array to get the left part
    # Combining to form a full circle from a quarter image and resizing the 700x700 to 200x200 image
    a_full=resize(np.concatenate((al,ar), axis=1), (heg, wid), anti_aliasing=True)
    return a_full

def illustrate(imgs,full=False,vmin=0,vmax=1.1):
    samp = imgs.clone().detach()
#     vmax = torch.max(torch.flatten(samp))
#     vmin = torch.min(torch.flatten(samp))
    fig, axs = plt.subplots(nrows=2, ncols=4, sharex=False, figsize=(18, 18))
    if vmin==-np.inf or vmax==np.inf:
        if full:
            vmax = torch.max(complete(samp[0,0,:,:,:]))
            vmin = torch.min(complete(samp[0,0,:,:,:]))
        else:
            vmax = torch.max(samp[0,0,:,:,:])
            vmin = torch.min(samp[0,0,:,:,:])
    for t in range(8):
        axs[t//4][t%4].set_title('t = ' + str(t))
        if full:
            hd = axs[t//4][t%4].imshow(complete(samp[0,0,t,:,:]),vmin=vmin, vmax=vmax)
        else:
            hd = axs[t//4][t%4].imshow(samp[0,0,t,:,:],vmin=vmin, vmax=vmax)
#         divider = make_axes_locatable(axs[t//4][t%4])
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         fig.colorbar(hd,cax=cax)
    cbar = fig.colorbar(hd, ax=axs.ravel().tolist(), shrink=0.95)
    cbar.set_ticks(np.arange(vmin, vmax, (vmax-vmin)/10))
    fig.suptitle('Generated dynamics, ' + str(t) + ' consecutive frames')
    plt.rcParams.update({'font.size': 10})
#     plt.tight_layout()
    plt.show()

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
    
datapath = '/mnt/shared_b/data/hydro_simulations/data/'
real_label = 1.
fake_label = 0.

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
    
def validate(testfiles,netD,netG,dep=8,batchsize=2,seed=0,img_size=320, \
             device="cpu",sigmoid_on=False,testfile_num=100):
#     filenum = len(testfiles)
    batchsize = min(testfile_num,batchsize)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # set the model in eval mode
    netD.eval()
    netG.eval()
    eval_score = 0; nrmse = 0; nl1err = 0; nlinferr = 0
    # evaluate on validation set
    fileind = 0
    criterion = nn.BCELoss() if sigmoid_on else nn.BCEWithLogitsLoss()
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
                    noise[bfile,0,:,:,:] = noise_generate(dyn[bfile,0,:,:,:],mode='const')
    #             for t in range(dep): # different noise for each frame when using a 'for' loop
    #                 noise[bfile,0,t,:,:] = noise_generate(dyn[bfile,0,t,:,:], mode='linear')
                sim.close()
                bfile += 1
            fileind += batchsize
            batch_step += 1

            dyn = torch.tensor(dyn).to(torch.float); noise = torch.tensor(noise).to(torch.float)
            real_cpu = dyn.to(device)
            b_size = real_cpu.size(0)

            ## Test with all-real batch
            Dx1_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            Dx1 = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(Dx1, Dx1_label)

            ## Test with all-fake batch
            fake = netG(noise + real_cpu)
            DGz1_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
            # Classify all fake batch with D
            DGz1 = netD(fake).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(DGz1, DGz1_label)

            eval_score = eval_score + (errD_real + errD_fake)
            nrmse      = nrmse      + aver_mse(fake,real_cpu)  * fake.shape[0]
            nl1err     = nl1err     + aver_l1(fake,real_cpu)   * fake.shape[0]
#             nlinferr   = nlinferr   + aver_linf(fake,real_cpu) * fake.shape[0]
    
    # set the model back to training mode
    netD.train()
    netG.train()
    return eval_score/(batch_step*batchsize), nrmse/(batch_step*batchsize), nl1err/(batch_step*batchsize) #, nlinferr/filenum

import importlib
import logging
import os
import shutil
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

plt.ioff()
plt.switch_backend('agg')


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def remove_halo(patch, index, shape, patch_halo):
    """
    Remove `pad_width` voxels around the edges of a given patch.
    """
    assert len(patch_halo) == 3

    def _new_slices(slicing, max_size, pad):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad
            i_start = slicing.start + pad

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad if pad != 0 else 1
            i_stop = slicing.stop - pad

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, patch.shape[0])

    p_z, i_z = _new_slices(i_z, D, patch_halo[0])
    p_y, i_y = _new_slices(i_y, H, patch_halo[1])
    p_x, i_x = _new_slices(i_x, W, patch_halo[2])

    patch_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return patch[patch_index], index


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.
        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def _find_masks(batch, min_size=10):
    """Center the z-slice in the 'middle' of a given instance, given a batch of instances
    Args:
        batch (ndarray): 5d numpy tensor (NCDHW)
    """
    result = []
    for b in batch:
        assert b.shape[0] == 1
        patch = b[0]
        z_sum = patch.sum(axis=(1, 2))
        coords = np.where(z_sum > min_size)[0]
        if len(coords) > 0:
            ind = coords[len(coords) // 2]
            result.append(b[:, ind:ind + 1, ...])
        else:
            ind = b.shape[1] // 2
            result.append(b[:, ind:ind + 1, ...])

    return np.stack(result, axis=0)


def get_tensorboard_formatter(config):
    if config is None:
        return DefaultTensorboardFormatter()

    class_name = config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**config)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays
    Args:
        inputs (iteable of torch.Tensor): torch tensor
    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def create_optimizer(optimizer_config, model):
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


def create_sample_plotter(sample_plotter_config):
    if sample_plotter_config is None:
        return None
    class_name = sample_plotter_config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**sample_plotter_config)