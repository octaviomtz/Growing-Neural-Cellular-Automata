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

import hydra
from omegaconf import DictConfig

# FUNCTIONS PLOT
def visualize_batch(x0, x):
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
    plt.show()

def plot_loss(loss_log, epochs=2000):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log_default), '.', alpha=0.1)
    plt.plot(np.log10(loss_log), '.', alpha=0.1, c='r')
    plt.ylim([-5, np.max(loss_log)])
    plt.xlim([0, epochs])
    plt.show()

def load_1ch_py_array(array_path):
    array = np.load(array_path)
    array = array.f.arr_0
    array = np.expand_dims(array,-1)
    array = np.repeat(array,2,-1)
    array[...,1] = array[...,0]>0 
    return array

# LOAD DEFAULT RESULTS
loss_log_default = np.load('data/loss_scale_growth=1_ep=10k.npy')
grow_sel_max_default = np.load('data/grow_max_scale_growth=1_ep=10k.npy')
mse_recons_default = np.load('data/mse_recons_default_ep=10k.npy')
loss_log_default2 = np.load('data/loss_scale_growth=1_ep=2k.npy')
grow_sel_max_default2 = np.load('data/grow_max_scale_growth=1_ep=2k.npy')
mse_recons_default2 = np.load('data/mse_recons_default_ep=2k.npy')

@hydra.main(config_path="config", config_name="config")
def cell_auto_config(cfg: DictConfig):
    seed = make_seed_1ch((h, w), cfg.CHANNEL_N)
    pool = SamplePool(x=np.repeat(seed[None, ...], cfg.POOL_SIZE, 0))

    ca = CAModel(cfg.CHANNEL_N, cfg.CELL_FIRE_RATE, cfg.device, scale_growth=cfg.SCALE_GROWTH).to(cfg.device)
    ca.load_state_dict(torch.load(cfg.model_path))

    optimizer = optim.Adam(ca.parameters(), lr=cfg.lr, betas=cfg.betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, cfg.lr_gamma)

    return pool, ca, optimizer, scheduler

@hydra.main(config_path="config", config_name="config")
def config_target(target_img, cfg: DictConfig):
    p = cfg.TARGET_PADDING
    pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
    h, w = pad_target.shape[:2]
    pad_target = np.expand_dims(pad_target, axis=0)
    pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(cfg.device)
    return pad_target

def train(ca, x, target, steps, optimizer, scheduler):
    x = ca(x, steps=steps)
    loss = F.mse_loss(x[:, :, :, :2], target)
    optimizer.zero_grad()
    loss.backward()
    ca.normalize_grads()
    optimizer.step()
    scheduler.step()
    return x, loss

def load_baselines():
    loss_log_base = np.load('data/loss_scale_growth=1_ep=10k.npy')
    grow_sel_max_base = np.load('data/grow_max_scale_growth=1_ep=10k.npy')
    mse_recons_base = np.load('data/mse_recons_default_ep=10k.npy')
    loss_log_base2 = np.load('data/loss_scale_growth=1_ep=2k.npy')
    grow_sel_max_base2 = np.load('data/grow_max_scale_growth=1_ep=2k.npy')
    mse_recons_base2 = np.load('data/mse_recons_default_ep=2k.npy')
    return loss_log_base, grow_sel_max_base, mse_recons_base, loss_log_base2, grow_sel_max_base2, mse_recons_base2, 

def main_train():
    loss_base, max_base, mse_base, loss_base2, max_base2, mse_base2 =  load_baselines()
    target_img = load_1ch_py_array('data/lesion2.npz')
    pad_target = config_target(target_img)




if __name__ == "__main__":
    main_train()