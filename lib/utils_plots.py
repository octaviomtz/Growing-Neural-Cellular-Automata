import numpy as np
import matplotlib.pyplot as plt

from lib.utils_vis import SamplePool, to_alpha_1ch, to_rgb_1ch

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


def plot_loss(loss_log, loss_log_base, SCALE_GROWTH, epochs=2000, save=True):
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
        plt.savefig('loss_training.png')

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

def plot_lesion_growing(grow_sel, target_img, ITER_SAVE, save=True):
    fig, ax = plt.subplots(5,6, figsize=(18,12))
    for i in range(ITER_SAVE):
        ax.flat[i].imshow(grow_sel[i], vmin=0, vmax=1)
        ax.flat[i].axis('off')
    ax.flat[0].imshow(target_img[...,0], vmin=0, vmax=1)
    fig.tight_layout()
    if save:
        plt.savefig('lesion_growing.png')

def load_baselines(path):
    loss_base = np.load(f'{path}/loss_scale_growth=1_ep=10k.npy')
    max_base = np.load(f'{path}/grow_max_scale_growth=1_ep=10k.npy')
    mse_base = np.load(f'{path}/mse_recons_default_ep=10k.npy')
    loss_base2 = np.load(f'{path}/loss_scale_growth=1_ep=2k.npy')
    max_base2 = np.load(f'{path}/grow_max_scale_growth=1_ep=2k.npy')
    mse_base2 = np.load(f'{path}/mse_recons_default_ep=2k.npy')
    return loss_base, max_base, mse_base, loss_base2, max_base2, mse_base2, 