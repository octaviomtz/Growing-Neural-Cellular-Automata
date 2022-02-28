#%% IMPORTS
import time
import imageio
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.morphology import remove_small_objects
from IPython.display import clear_output
import monai
from lib.CAModel import CAModel
from tqdm import tqdm
from lib.utils_plots import visualize_batch, plot_loss_max_intensity_and_mse, plot_lesion_growing, load_baselines, make_seed_1ch, plot_seeds
from lib.utils_monai import (load_COVID19_v2,
                            load_synthetic_lesions,
                            load_scans,
                            load_individual_lesions,
                            load_synthetic_texture)
from lib.utils_superpixels import (
    superpixels,
    make_list_of_targets_and_seeds,
    fig_superpixels_only_lesions,
    select_lesions_match_conditions2,
    boundaries_superpixels,
    how_large_is_each_segment
)



#%% FUNCTIONS
def prepare_seed(target, this_seed, device, num_channels = 16, pool_size = 1024):
    # prepare seed
    height, width, _ = np.shape(target)
    seed = np.zeros([height, width, num_channels], np.float32)
    for i in range(num_channels-1):
        seed[..., i+1] = this_seed
    return seed

def config_cellular_automata(orig_dir, CHANNEL_N, CELL_FIRE_RATE, device, SCALE_GROWTH, model_path, lr, betas_0, betas_1, lr_gamma):
    ca = CAModel(CHANNEL_N, CELL_FIRE_RATE, device, scale_growth=SCALE_GROWTH).to(device)
    # ca.load_state_dict(torch.load(f'{orig_dir}/{model_path}'))

    optimizer = optim.Adam(ca.parameters(), lr=lr, betas=(betas_0, betas_1))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

    return ca, optimizer, scheduler

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
    # ca.normalize_grads()
    optimizer.step()
    scheduler.step()
    return x, loss

#%% 
cfg_data_data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train'
cfg_data_SCAN_NAME = 'volume-covid19-A-0014'
cfg_data_BATCH_SIZE = 1
cfg_data_path_single_lesions = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/'
cfg_data_path_texture = '/content/drive/My Drive/Datasets/covid19/results/cea_synthesis/patient0/'
cfg_loop_SKIP_LESIONS = -1
cfg_loop_ONLY_ONE_SLICE = True
cfg_data_SLICE = 22
cfg_loop_TRESH_PLOT = 20
cfg_seed_SEED_VALUE = 0.19
#%%
images, labels, keys, files_scans = load_COVID19_v2(cfg_data_data_folder, cfg_data_SCAN_NAME)
name_prefix = load_synthetic_lesions(files_scans, keys, cfg_data_BATCH_SIZE)
scan, scan_mask = load_scans(files_scans, keys, cfg_data_BATCH_SIZE, cfg_data_SCAN_NAME)
path_single_lesions = f'{cfg_data_path_single_lesions}{cfg_data_SCAN_NAME}_ct/'
loader_lesions = load_individual_lesions(path_single_lesions, cfg_data_BATCH_SIZE)
texture = load_synthetic_texture(cfg_data_path_texture)
print(scan.shape, scan_mask.shape, texture.shape)

#%% SUPERPIXELS
mask_sizes=[]
cluster_sizes = []
targets_all = []
flag_slice_found = False
for idx_mini_batch,mini_batch in enumerate(loader_lesions):
    if idx_mini_batch < cfg_loop_SKIP_LESIONS:continue #resume incomplete reconstructions

    img = mini_batch['image'].numpy()
    mask = mini_batch['label'].numpy()
    mask = remove_small_objects(mask, 20)
    mask_sizes.append([idx_mini_batch, np.sum(mask)])
    name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.npy')[0].split('19-')[-1]
    img_lesion = img*mask

    # if 2nd argument is provided then only analyze that slice
    if cfg_loop_ONLY_ONE_SLICE: 
        slice_used = int(name_prefix.split('_')[-1])
        if slice_used != int(cfg_data_SLICE): continue
        else: flag_slice_found = True

    mask_slic, boundaries, segments, numSegments = boundaries_superpixels(img[0], mask[0])
    segments_sizes = how_large_is_each_segment(segments)

    print(f'img = {np.shape(img)}')

    tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
    targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=cfg_seed_SEED_VALUE, seed_method='max')
    targets_all.append(len(targets))

    coords_big = [int(i) for i in name_prefix.split('_')[1:]]
    fig_superpixels_only_lesions('./', name_prefix, scan, scan_mask, img, mask_slic, boundaries, segments, segments_sizes, coords_big, cfg_loop_TRESH_PLOT, idx_mini_batch, numSegments, save=False)
    if flag_slice_found: break

if flag_slice_found:
    plot_seeds(targets,seeds, save=False)
    for i in targets:
        print(f'targets_shape = {i.shape}')
else:
    print('SLICE HAS NO LESIONS')

#%%
mask_slic, boundaries, segments, numSegments = boundaries_superpixels(img[0], mask[0])
len(segments), numSegments
tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=cfg_seed_SEED_VALUE, seed_method='max')
len(targets), targets[0].shape




#%%
path_orig = './'
extra_text = '_volume-covid19-A-0014_34_2'
max_2k, mse_2k, train_loss_2k, max_10k, mse_10k, train_loss_10k = load_baselines(path_orig, extra_text)
plt.plot(train_loss_2k)
#%%
#%%

#%% FUNCTIONS
def load_emoji(index, path="data/emoji.png"):
    im = imageio.imread(path)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
    emoji /= 255.0
    return emoji

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

#%% HYPERPARAMS
device = torch.device("cuda:0")
model_path = "models/remaster_1.pth"

CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 0   # Number of pixels used to pad the target image border
TARGET_SIZE = 40

lr = 2e-3
lr_gamma = 0.9999
betas = (0.5, 0.5)
EPOCHS = 200

BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5
SCALE_GROWTH = .1

#%% LOAD DEFAULT RESULTS
path_files = 'data/lesion2/'
loss_log_default = np.load('data/lesion2/loss_scale_growth=1_ep=10k.npy')
grow_sel_max_default = np.load('data/lesion2/grow_max_scale_growth=1_ep=10k.npy')
mse_recons_default = np.load('data/lesion2/mse_recons_default_ep=10k.npy')
loss_log_default2 = np.load('data/lesion2/loss_scale_growth=1_ep=2k.npy')
grow_sel_max_default2 = np.load('data/lesion2/grow_max_scale_growth=1_ep=2k.npy')
mse_recons_default2 = np.load('data/lesion2/mse_recons_default_ep=2k.npy')

#%%
target_img = load_1ch_py_array('data/lesion2/lesion.npz')
print(np.shape(target_img))
# plt.figure(figsize=(4,4))
# plt.imshow(np.squeeze(to_rgb_1ch(target_img)))

#%%
print(len(loss_log_default))

# %% SEED AND nCA
p = TARGET_PADDING
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

seed = make_seed_1ch((h, w), CHANNEL_N)
# pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))
#%%
ca = CAModel(CHANNEL_N, CELL_FIRE_RATE, device, scale_growth=SCALE_GROWTH).to(device)
# ca.load_state_dict(torch.load(model_path))

optimizer = optim.Adam(ca.parameters(), lr=lr, betas=betas)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

# %% RUN nCA
loss_log = []

def train(x, target, steps, optimizer, scheduler):
    x = ca(x, steps=steps)
    loss = F.mse_loss(x[:, :, :, :2], target)
    optimizer.zero_grad()
    loss.backward()
    # ca.normalize_grads()
    optimizer.step()
    scheduler.step()
    return x, loss

for i in range(EPOCHS+1):
    
    x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    x0 = torch.from_numpy(x0.astype(np.float32)).to(device)

    x, loss = train(x0, pad_target, np.random.randint(64,96), optimizer, scheduler)
    
    step_i = len(loss_log)
    loss_log.append(loss.item())
    
    if step_i%100 == 0:
        clear_output()
        print(step_i, "loss =", loss.item())
        visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy())
        plot_loss(loss_log)
        # torch.save(ca.state_dict(), model_path)

# %% GROW TRAINED nCA
grow = torch.tensor(seed).unsqueeze(0).to(device)
grow_sel = []
ITER_GROW = 60
ITER_SAVE = 30
grow_sel_max = []
with torch.no_grad():
    for i in range(ITER_GROW):
        grow = ca(grow)
        if i % (ITER_GROW // ITER_SAVE) == 0:
            grow_img = grow.detach().cpu().numpy()
            grow_img = grow_img[0,...,:1]
            grow_img = np.squeeze(np.clip(grow_img,0,1))
            grow_sel.append(grow_img)
            grow_sel_max.append(np.max(grow_img))
# %% PLOT GROWING LESIONS
fig, ax = plt.subplots(5,6, figsize=(18,12))
for i in range(ITER_SAVE):
    ax.flat[i].imshow(grow_sel[i], vmin=0, vmax=1)
    ax.flat[i].axis('off')
fig.tight_layout()

# %% MSE RECONSTRUCTION
target_padded = pad_target.detach().cpu().numpy()[0,...,0]
mse_recons = []
for i in range(ITER_SAVE):
    mse_recons.append(np.mean((grow_sel[i] - target_padded)**2))

# %% PLOT MAX INTENSITY AND MSE
plt.style.use("Solarize_Light2")
fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(grow_sel_max_default, label='(10k) scale = 1', alpha=.3)
ax[0].plot(grow_sel_max_default2, label='(2k) scale = 1', alpha=.3)
ax[0].plot(grow_sel_max, label=f'scale={SCALE_GROWTH:.02f}')
ax[0].legend()
ax[1].semilogy(mse_recons_default, label='(10k) scale = 1', alpha=.3)
ax[1].semilogy(mse_recons_default2, label='(2k) scale = 1', alpha=.3)
ax[1].semilogy(mse_recons, label=f'scale={SCALE_GROWTH:.02f}')
ax[1].legend()

########### finish
#%%
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# %%

# np.save('data/loss_scale_growth=1_ep=2k.npy', loss_log)
# np.save('data/grow_max_scale_growth=1_ep=2k.npy', grow_sel_max)
# np.save('data/mse_recons_default_ep=2k.npy', mse_recons)

#%%
plt.figure(figsize=(10, 4))
plt.title('Loss history (log10)')
# plt.plot(np.log10(loss_log_default), '.', alpha=0.1)
plt.plot(np.log10(loss_log), '.', alpha=0.1, c='b')
# plt.plot(movingaverage(np.log10(loss_log),5))
plt.plot()
plt.ylim([-6, -1])
plt.xlim([0, 2000])
plt.show()
# %%
len(loss_log)
# %%
EPOCHS = 200
# %%
