import os
import numpy as np
import matplotlib.pyplot as plt

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


def plot_loss(loss_log, SCALE_GROWTH, loss_log_base=-1, epochs=2000, save=True, text=''):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    if loss_log_base != -1:
        plt.plot(np.log10(loss_log_base), '.', alpha=0.1, label='base scale=1')
    plt.plot(np.log10(loss_log), '.', alpha=0.1, c='r', label=f'scale={SCALE_GROWTH:.02f}')
    plt.ylim([-5, np.max(loss_log)])
    plt.xlim([0, epochs])
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('log10(MSE)')
    if save==True:
        plt.savefig(f'loss_training{text}.png')

def plot_max_intensity_and_mse(grow_max, mse_recons, SCALE_GROWTH, max_base=-1, max_base2=-1, mse_base=-1, mse_base2=-1, save=True, text=''):
    # %% PLOT MAX INTENSITY AND MSE
    plt.style.use("Solarize_Light2")
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    if max_base!=-1: ax[0].plot(max_base, label='(10k) scale = 1', alpha=.3)
    if max_base2!=-1: ax[0].plot(max_base2, label='(2k) scale = 1', alpha=.3)
    ax[0].plot(grow_max, label=f'scale={SCALE_GROWTH:.02f}')
    ax[0].legend(loc = 'lower left')
    ax[0].set_xlabel('reconstruction epochs')
    ax[0].set_ylabel('max intensity')
    if mse_base!=-1: ax[1].semilogy(mse_base, label='(10k) scale = 1', alpha=.3)
    if mse_base2!=-1: ax[1].semilogy(mse_base2, label='(2k) scale = 1', alpha=.3)
    ax[1].semilogy(mse_recons, label=f'scale={SCALE_GROWTH:.02f}')
    ax[1].legend(loc = 'lower left')
    ax[1].set_xlabel('reconstruction epochs')
    ax[1].set_ylabel('MSE')
    fig.tight_layout()
    if save:
        plt.savefig(f'max_intensity_and_mse{text}.png')

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
            outputs.append([-1])
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

    