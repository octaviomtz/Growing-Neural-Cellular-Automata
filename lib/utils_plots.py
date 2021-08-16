import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from lib.utils_vis import SamplePool, to_alpha_1ch, to_rgb_1ch

def visualize_batch(x0, x, save=True, text=''):
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
        plt.savefig(f'visualize_batch{text}.png')


def plot_loss(loss_log, SCALE_GROWTH, loss_log_base=-1, epochs=2000, save=True, save_wandb=False, text=''):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log_base), '.', alpha=0.1, label='base scale=1')
    plt.plot(np.log10(loss_log), '.', alpha=0.1, c='r', label=f'scale={SCALE_GROWTH:.02f}')
    plt.ylim([-5, np.max(loss_log)])
    plt.xlim([0, epochs])
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('log10(MSE)')
    if save==True:
        plt.savefig(f'loss_training{text}.png')
    if save_wandb:
        wandb.log({f'loss_training{text}.png': wandb.Image(plt)})

def plot_loss_max_intensity_and_mse(loss_log, loss_log_base, SCALE_GROWTH, SCALE_GROWTH_SYN, grow_max, mse_recons, max_base, max_base2, mse_base, mse_base2, epochs=2000, save=True, save_wandb=False, text=''):
    plt.style.use("Solarize_Light2")
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax1.plot(np.log10(loss_log_base), '.', alpha=0.1, label='base scale=1')
    ax1.plot(np.log10(loss_log), '.', alpha=0.1, c='r', label=f'scale={SCALE_GROWTH:.02f}')
    ax1.set_ylim([-5, np.max(loss_log)])
    ax1.set_xlim([0, epochs])
    ax1.legend()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('log10(MSE)')
    ax2.plot(max_base, label='(10k) scale = 1', alpha=.5)
    ax2.plot(max_base2, label='(2k) scale = 1', alpha=.5)
    ax2.plot(grow_max, label=f'scales={SCALE_GROWTH:.02f}_{SCALE_GROWTH_SYN:.02f}')
    ax2.legend(loc = 'lower left')
    ax2.set_xlabel('reconstruction epochs (x2)')
    ax2.set_ylabel('max intensity')
    ax3.semilogy(mse_base, label='(10k) scale = 1', alpha=.5)
    ax3.semilogy(mse_base2, label='(2k) scale = 1', alpha=.5)
    ax3.semilogy(mse_recons, label=f'scales={SCALE_GROWTH:.02f}_{SCALE_GROWTH_SYN:.02f}')
    ax3.legend(loc = 'lower left')
    ax3.set_xlabel('reconstruction epochs (x2)')
    ax3.set_ylabel('MSE')
    fig.tight_layout()
    if save:
        plt.savefig(f'train_loss_and_synthesis{text}.png')
    if save_wandb:
        wandb.log({f'train_loss_and_synthesis{text}.png': wandb.Image(plt)})

def plot_max_intensity_and_mse(grow_max, mse_recons, SCALE_GROWTH, SCALE_GROWTH_SYN, max_base=-1, max_base2=-1, mse_base=-1, mse_base2=-1, save=True, save_wandb=False, text=''):
    # %% PLOT MAX INTENSITY AND MSE
    plt.style.use("Solarize_Light2")
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(max_base, label='(10k) scale = 1', alpha=.3)
    ax[0].plot(max_base2, label='(2k) scale = 1', alpha=.3)
    ax[0].plot(grow_max, label=f'scales={SCALE_GROWTH:.02f}_{SCALE_GROWTH_SYN:.02f}')
    ax[0].legend(loc = 'lower left')
    ax[0].set_xlabel('reconstruction epochs')
    ax[0].set_ylabel('max intensity')
    ax[1].semilogy(mse_base, label='(10k) scale = 1', alpha=.3)
    ax[1].semilogy(mse_base2, label='(2k) scale = 1', alpha=.3)
    ax[1].semilogy(mse_recons, label=f'scales={SCALE_GROWTH:.02f}_{SCALE_GROWTH_SYN:.02f}')
    ax[1].legend(loc = 'lower left')
    ax[1].set_xlabel('reconstruction epochs')
    ax[1].set_ylabel('MSE')
    fig.tight_layout()
    if save:
        plt.savefig(f'max_intensity_and_mse{text}.png')
    if save_wandb:
        wandb.log({f'max_intensity_and_mse{text}.png': wandb.Image(plt)})

def plot_lesion_growing(grow_sel, target_img, ITER_SAVE, save=True, text=''):
    fig, ax = plt.subplots(5,6, figsize=(18,12))
    for i in range(ITER_SAVE):
        ax.flat[i].imshow(grow_sel[i], vmin=0, vmax=1)
        ax.flat[i].axis('off')
    ax.flat[0].imshow(target_img[...,0], vmin=0, vmax=1)
    fig.tight_layout()
    if save:
        plt.savefig(f'lesion_growing{text}.png')

def load_baselines(path_orig, extra_text, path='outputs/baselines/'):
    files = os.listdir(f'{path_orig}/{path}')
    outputs = []
    for key in ['max_syn_SG=1_ep=2k', 'mse_syn_SG=1_ep=2k', 'train_loss_SG=1_ep=2k', 'max_syn_SG=1_ep=10k', 'mse_syn_SG=1_ep=10k', 'train_loss_SG=1_ep=10k']:
        file = f'{key}{extra_text}.npy'
        if file in files:
            outputs.append(np.load(f'{path_orig}/{path}{file}'))
        else:
            outputs.append([.001, .001])
    return outputs 

def make_seed_1ch(shape, n_channels):
    seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
    seed[shape[0]//2, shape[1]//2, 1:] = 1.0
    return seed

def plot_seeds(targets,seeds, save=True):
    fig, ax = plt.subplots(2,2)
    for idx, (t,s) in enumerate(zip(targets,seeds)):
        # print(f'target={np.shape(t)}{np.unique(t[...,1])} seed={np.shape(s)}{np.unique(s)}')
        ax.flat[idx].imshow(t[...,1])
        ax.flat[idx].imshow(s, alpha=.3)
    if save:
        plt.savefig('seeds.png')

def save_cell_auto_reconstruction_vars(grow_sel, coord, mask, losses, name_prefix, idx_lesion):
    outs_float = np.asarray(grow_sel)
    np.savez_compressed(f'{name_prefix}_lesion_{idx_lesion:02d}.npz', outs_float)
    np.save(f'{name_prefix}_coords_{idx_lesion:02d}.npy', coord)
    np.savez_compressed(f'{name_prefix}_mask_{idx_lesion:02d}.npz', mask)
    np.save(f'{name_prefix}_loss_{idx_lesion:02d}.npy', losses)