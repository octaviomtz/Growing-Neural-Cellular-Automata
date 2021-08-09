# IMPORTS
import time
import imageio

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from IPython.display import clear_output

from lib.CAModel import CAModel
from lib.utils_vis import SamplePool, to_alpha_1ch, to_rgb_1ch, get_living_mask, make_seed_1ch, make_circle_masks

from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import logging

# FUNCTIONS PLOT
def visualize_batch(x0, x, save=True):
    plt.style.use("Solarize_Light2")
    vis0 = to_rgb_1ch(x0)
    vis1 = to_rgb_1ch(x)
    # vis0 = x0[...,0]
    # vis1 = x[...,0]
    print('batch (before/after):')
    plt.figure(figsize=[15,5])
    for i in range(x0.shape[0]):
        plt.subplot(2,x0.shape[0],i+1)
        plt.imshow(np.squeeze(vis0[i]))
        plt.axis('off')
    for i in range(x0.shape[0]):
        plt.subplot(2,x0.shape[0],i+1+x0.shape[0])
        plt.imshow(np.squeeze(vis1[i]))
        plt.axis('off')
    if save==True:
        plt.savefig('visualize_batch.png')

def plot_loss(loss_log, loss_log_base, epochs=2000, save=True):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log_base), '.', alpha=0.1, label='base scale=1')
    plt.plot(np.log10(loss_log), '.', alpha=0.1, c='r', label='new')
    plt.ylim([-5, np.max(loss_log)])
    plt.xlim([0, epochs])
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('log10(MSE)')
    if save==True:
        plt.savefig('loss_training.png')

def load_1ch_py_array(dir,array_path):
    array_path = f'{dir}/{array_path}'
    array = np.load(array_path)
    array = array.f.arr_0
    array = np.expand_dims(array,-1)
    array = np.repeat(array,2,-1)
    array[...,1] = array[...,0]>0 
    return array

# LOAD DEFAULT RESULTS
loss_base = np.load('data/loss_scale_growth=1_ep=10k.npy')
max_base = np.load('data/grow_max_scale_growth=1_ep=10k.npy')
mse_base = np.load('data/mse_recons_default_ep=10k.npy')
loss_base2 = np.load('data/loss_scale_growth=1_ep=2k.npy')
max_base2 = np.load('data/grow_max_scale_growth=1_ep=2k.npy')
mse_base2 = np.load('data/mse_recons_default_ep=2k.npy')

def config_cellular_automata(orig_dir, height_width, CHANNEL_N, CELL_FIRE_RATE, device, SCALE_GROWTH, model_path, lr, betas_0, betas_1, lr_gamma):
    h, w = height_width
    seed = make_seed_1ch((h, w), CHANNEL_N)
    # pool = SamplePool(x=np.repeat(seed[None, ...], cfg.POOL_SIZE, 0))

    ca = CAModel(CHANNEL_N, CELL_FIRE_RATE, device, scale_growth=SCALE_GROWTH).to(device)
    ca.load_state_dict(torch.load(f'{orig_dir}/{model_path}'))

    optimizer = optim.Adam(ca.parameters(), lr=lr, betas=(betas_0, betas_1))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

    return ca, optimizer, scheduler, seed

def pad_target_func(target_img, padding, device):
    p = padding
    pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
    h, w = pad_target.shape[:2]
    height_width = [h, w]
    pad_target = np.expand_dims(pad_target, axis=0)
    pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(torch.device(device))
    return pad_target, height_width   

def train(ca, x, target, steps, optimizer, scheduler):
    """Runs cellular automata (nCA), backpropagates and updates the weights

    Args:
        ca ([type]): cellular automata model
        x ([type]): image to update with nCA
        target ([type]): target image
        steps ([type]): how many times we update without feedback (1)
        optimizer (pytorch optim): 
        scheduler (pytorch scheduler): 

    Returns:
        x [type]: reconstructed image
        loss [numpy]: 
    """
    x = ca(x, steps=steps)
    loss = F.mse_loss(x[:, :, :, :2], target)
    optimizer.zero_grad()
    loss.backward()
    ca.normalize_grads()
    optimizer.step()
    scheduler.step()
    return x, loss

def load_baselines(dir):
    loss_base = np.load(f'{dir}/data/loss_scale_growth=1_ep=10k.npy')
    max_base = np.load(f'{dir}/data/grow_max_scale_growth=1_ep=10k.npy')
    mse_base = np.load(f'{dir}/data/mse_recons_default_ep=10k.npy')
    loss_base2 = np.load(f'{dir}/data/loss_scale_growth=1_ep=2k.npy')
    max_base2 = np.load(f'{dir}/data/grow_max_scale_growth=1_ep=2k.npy')
    mse_base2 = np.load(f'{dir}/data/mse_recons_default_ep=2k.npy')
    return loss_base, max_base, mse_base, loss_base2, max_base2, mse_base2, 

def plot_max_intensity_and_mse(max_base, max_base2, grow_max, mse_base, mse_base2, mse_recons, SCALE_GROWTH, save=True):
    # %% PLOT MAX INTENSITY AND MSE
    plt.style.use("Solarize_Light2")
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(max_base, label='(10k) scale = 1', alpha=.3)
    ax[0].plot(max_base2, label='(2k) scale = 1', alpha=.3)
    ax[0].plot(grow_max, label=f'scale={SCALE_GROWTH:.02f}')
    ax[0].legend(loc = 'lower left')
    ax[0].set_xlabel('reconstruction epochs')
    ax[0].set_ylabel('max intensity')
    ax[1].semilogy(mse_base, label='(10k) scale = 1', alpha=.3)
    ax[1].semilogy(mse_base2, label='(2k) scale = 1', alpha=.3)
    ax[1].semilogy(mse_recons, label=f'scale={SCALE_GROWTH:.02f}')
    ax[1].legend(loc = 'lower left')
    ax[1].set_xlabel('reconstruction epochs')
    ax[1].set_ylabel('MSE')
    fig.tight_layout()
    if save:
        plt.savefig('max_intensity_and_mse.png')

@hydra.main(config_path="config", config_name="config.yaml")
def main_train(cfg: DictConfig):
    log = logging.getLogger(__name__)
    
    orig_work_dir = hydra.utils.get_original_cwd()
    loss_base, max_base, mse_base, loss_base2, max_base2, mse_base2 =  load_baselines(orig_work_dir)
    target_img = load_1ch_py_array(orig_work_dir,'data/lesion2.npz')
    pad_target, height_width = pad_target_func(target_img, cfg.TARGET_PADDING, cfg.device)
    ca, optimizer, scheduler, seed = config_cellular_automata(orig_work_dir, height_width, cfg.CHANNEL_N, cfg.CELL_FIRE_RATE, cfg.device, cfg.SCALE_GROWTH,  cfg.model_path, cfg.lr, cfg.betas_0, cfg.betas_1, cfg.lr_gamma)
    
    loss_log = []
    for i in tqdm(range(cfg.EPOCHS+1)):
    
        x0 = np.repeat(seed[None, ...], cfg.BATCH_SIZE, 0)
        x0 = torch.from_numpy(x0.astype(np.float32)).to(cfg.device)

        x, loss = train(ca, x0, pad_target, np.random.randint(64,96), optimizer, scheduler)
        loss_log.append(loss.item())
        log.info(f"loss = {loss.item()}")

    visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy())
    plot_loss(loss_log, loss_base)

    grow = torch.tensor(seed).unsqueeze(0).to(cfg.device)
    grow_sel = []
    grow_max = []
    mse_recons = []
    target_padded = pad_target.detach().cpu().numpy()[0,...,0]
    with torch.no_grad():
        for i in range(cfg.ITER_GROW):
            grow = ca(grow)
            if i % (cfg.ITER_GROW // cfg.ITER_SAVE) == 0:
                grow_img = grow.detach().cpu().numpy()
                grow_img = np.squeeze(np.clip(grow_img[0,...,:1],0,1))
                mse_recons.append(np.mean((grow_img - target_padded)**2))
                grow_sel.append(grow_img)
                grow_max.append(np.max(grow_img))
    

    plot_max_intensity_and_mse(max_base, max_base2, grow_max, mse_base, mse_base2, mse_recons, cfg.SCALE_GROWTH)
    print(pad_target.shape, height_width)

if __name__ == "__main__":
    main_train()