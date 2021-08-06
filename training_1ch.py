#%% IMPORTS
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

#%% FUNCTIONS
def load_emoji(index, path="data/emoji.png"):
    im = imageio.imread(path)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
    emoji /= 255.0
    return emoji

def visualize_batch(x0, x):
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

def plot_loss(loss_log):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.1)
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
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40

lr = 2e-3
lr_gamma = 0.9999
betas = (0.5, 0.5)
n_epoch = 80000

BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

# # TARGET_EMOJI = 0 #@param "🦎"

# EXPERIMENT_TYPE = "Growing"
# EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
# EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

# USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
# DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

#%%
target_img = load_1ch_py_array('data/lesion2.npz')
print(np.shape(target_img))
plt.figure(figsize=(4,4))
plt.imshow(np.squeeze(to_rgb_1ch(target_img)))

# %% SEED AND nCA
p = TARGET_PADDING
pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
h, w = pad_target.shape[:2]
pad_target = np.expand_dims(pad_target, axis=0)
pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

seed = make_seed_1ch((h, w), CHANNEL_N)
pool = SamplePool(x=np.repeat(seed[None, ...], POOL_SIZE, 0))

ca = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
ca.load_state_dict(torch.load(model_path))

optimizer = optim.Adam(ca.parameters(), lr=lr, betas=betas)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

# %% RUN nCA
loss_log = []

def train(x, target, steps, optimizer, scheduler):
    x = ca(x, steps=steps)
    loss = F.mse_loss(x[:, :, :, :2], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x, loss

for i in range(n_epoch+1):
    
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
with torch.no_grad():
    for i in range(ITER_GROW):
        grow = ca(grow)
        if i % (ITER_GROW // ITER_SAVE) == 0:
            grow_img = grow.detach().cpu().numpy()
            grow_img = grow_img[0,...,:1]
            grow_img = np.squeeze(np.clip(grow_img,0,1))
            grow_sel.append(grow_img)
# %% PLOT GROWING LESIONS
fig, ax = plt.subplots(5,6, figsize=(18,12))
for i in range(ITER_SAVE):
    ax.flat[i].imshow(grow_sel[i])
    ax.flat[i].axis('off')
fig.tight_layout()
# %%
