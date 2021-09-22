#%%
import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import glob
from copy import copy

from IPython.display import Image, HTML, clear_output
from tqdm import tqdm_notebook, tnrange

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import matplotlib.patches as patches
import torch
import torchvision.models as models

torch.set_default_tensor_type('torch.cuda.FloatTensor')

from lib.utils_texture_nca import (imread, np2pil, imwrite, imencode, im2url, imshow, tile2d, zoom, VideoWriter)
from lib.utils_texture_nca import calc_styles, to_nchw, style_loss

from lib.utils_lung_segmentation import find_closest_cluster, find_texture_relief

#%% title Minimalistic Neural CA
ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])

def perchannel_conv(x, filters):
  '''filters: [filter_n, h, w]'''
  b, ch, h, w = x.shape
  y = x.reshape(b*ch, 1, h, w)
  y = torch.nn.functional.pad(y, [1, 1, 1, 1], 'circular')
  y = torch.nn.functional.conv2d(y, filters[:,None])
  return y.reshape(b, -1, h, w)

def perception(x):
  filters = torch.stack([ident, sobel_x, sobel_x.T, lap])
  return perchannel_conv(x, filters)

class CA(torch.nn.Module):
  def __init__(self, chn=12, hidden_n=96):
    super().__init__()
    self.chn = chn
    self.w1 = torch.nn.Conv2d(chn*4, hidden_n, 1)
    self.w2 = torch.nn.Conv2d(hidden_n, chn, 1, bias=False)
    self.w2.weight.data.zero_()

  def forward(self, x, update_rate=0.5, noise=None):
    if noise is not None:
      x += torch.randn_like(x)*noise
    y = perception(x)
    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
    return x+y*udpate_mask

  def seed(self, n, sz=128):
    return torch.zeros(n, self.chn, sz, sz)

def to_rgb(x):
  return x[...,:3,:,:]+0.5

#%%
#@title Target image {vertical-output: true}
style_urls = [
  'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/bubbly/bubbly_0101.jpg',
  'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/chequered/chequered_0121.jpg'
]
style_imgs = [imread(url, max_size=128) for url in style_urls]
with torch.no_grad():
    target_styles = calc_styles(to_nchw(style_imgs))
imshow(np.hstack(style_imgs))

#%%
# print(style_imgs[0].shape)
texture_lungs = np.load('data/texture_lung_inpainted.npy')
texture_lesion = np.load('data/texture_lesion2_inpain.npy')
texture_lesion = texture_lesion[0]
print(texture_lungs.shape, texture_lesion.shape)
texture_lungs = np.moveaxis(texture_lungs,0,-1) 
# texture_lesion = np.moveaxis(texture_lesion,0,-1)
texture_lungs = texture_lungs[:128,:128,:]
texture_lesion = texture_lesion[:128,:128,:]
style_imgs = [texture_lungs, texture_lesion]
with torch.no_grad():
    target_styles = calc_styles(to_nchw(style_imgs))
imshow(np.hstack(style_imgs))

# %%
#@title setup training
ca = CA() 
opt = torch.optim.Adam(ca.parameters(), 1e-3)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], 0.3)
loss_log = []
with torch.no_grad():
  pool = ca.seed(1024)

# %% TRAINING WITH NOISE
#@title training loop {vertical-output: true}
for i in range(5000):
  with torch.no_grad():
    batch_idx = np.random.choice(len(pool), 4, replace=False)
    x = pool[batch_idx]
    if i%8 == 0:
      x[:1] = ca.seed(1)
    noise_mask = torch.tensor(batch_idx%2)
  step_n = np.random.randint(32, 96)
  for k in range(step_n):
    x = ca(x, noise=0.02*noise_mask.reshape(-1, 1, 1, 1))
  imgs = to_rgb(x)
  styles = calc_styles(imgs)
  overflow_loss = (x-x.clamp(-1.0, 1.0)).abs().sum()
  batch_targets = [g[noise_mask] for g in target_styles]
  loss = style_loss(styles, batch_targets)+overflow_loss
  with torch.no_grad():
    loss.backward()
    for p in ca.parameters():
      p.grad /= (p.grad.norm()+1e-8)   # normalize gradients 
    opt.step()
    opt.zero_grad()
    lr_sched.step()
    pool[batch_idx] = x                # update pool
    
    loss_log.append(loss.item())
    if i%32==0:
      clear_output(True)
      pl.plot(loss_log, '.', alpha=0.1)
      pl.yscale('log')
      pl.ylim(np.min(loss_log), loss_log[0])
      pl.show()
      imgs = to_rgb(x).permute([0, 2, 3, 1]).cpu()
      imshow(np.hstack(imgs))
    if i%10 == 0:
      print('\rstep_n:', len(loss_log),
        ' loss:', loss.item(), 
        ' lr:', lr_sched.get_lr()[0],
        'overflow:', overflow_loss.item(), end='')

#%%
torch.save(ca, 'data/ca_lungs_covid2.pt')

# %%
with VideoWriter('noise_ca_lungs_lesion_.mp4') as vid, torch.no_grad():
  noise = 0.5-0.5*torch.linspace(0, np.pi*2.0, 256).cos()
  noise *= 0.02
  x = torch.zeros(1, ca.chn, 128, 256)
  for k in tnrange(600, leave=False):
    step_n = 16
    frame_noise = noise if k<200 or k>400 else noise.max()-noise
    for i in range(step_n):
      x[:] = ca(x, noise=frame_noise)
      img = to_rgb(x[0]).permute(1, 2, 0).cpu()
    vid.add(zoom(img, 2))
vid.show(loop=True)
# %%
h, w = 1080//2, 1920//2
mask = PIL.Image.new('L', (w, h))
draw = PIL.ImageDraw.Draw(mask)
font = PIL.ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', 300)
draw  = PIL.ImageDraw.Draw(mask)
draw.text((20, 100), 'COVID', fill=255, font=font, align='center')
noise = torch.tensor(np.float32(mask)/255.0*0.02)
mask

#%%
h, w = 1080//2, 1920//2
with VideoWriter('covid_x2.mp4') as vid, torch.no_grad():
  x = torch.zeros(1, ca.chn, h, w)
  for k in tnrange(1000, leave=False):
    x[:] = ca(x, noise=noise)
  for k in tnrange(600, leave=False):
    for k in range(10):
      x[:] = ca(x, noise=noise)
    img = to_rgb(x[0]).permute(1, 2, 0).cpu()
    vid.add(img)
vid.show(loop=True)
# %%

#%% LOAD TRAINED MODEL
ca = torch.load('data/ca_lungs_covid2.pt')

#%%
texture_lesion = np.load('data/texture_lesion_inpainted.npy')
aa = texture_lesion[0,100:150,100:150]
labelled_boundaries, labelled, area_mod = find_texture_relief(aa)

fig, ax = pl.subplots(1,3, figsize=(12,8))
ax[0].imshow(aa)
ax[1].imshow(area_mod)
ax[2].imshow(labelled_boundaries)

#%%
LAB=15
# pl.imshow(labelled_boundaries==LAB)
growing_cluster2 = []
growing_cluster2.append((labelled==LAB)*area_mod)
for i in range(np.unique(labelled)[-1]-1):
    clus_number, clus_coord = find_closest_cluster(labelled_boundaries, LAB)
    labelled_boundaries[np.where(labelled_boundaries == clus_number)]=LAB
    labelled[np.where(labelled == clus_number)]=LAB
    labeles_joined = (labelled==LAB)
    growing_cluster2.append(labeles_joined*area_mod)
fig, ax = pl.subplots(6,6, figsize=(6,6))
for i in range(36):

    ax.flat[i].imshow(growing_cluster2[i])
    ax.flat[i].axis('off')

#%% JUST ONE CLUSTER
from scipy.ndimage import distance_transform_bf
aa = labelled==15
aa_x = distance_transform_bf(aa)
fig, ax = pl.subplots(1,2, figsize=(8,4))
ax[0].imshow(aa)
ax[1].imshow(aa_x)

#%%
pl.imshow(labelled)

#%%
masks_growing_cluster = []
for i in reversed(np.unique(aa_x)):
    masks_growing_cluster.append(aa_x>=i)
fig, ax = pl.subplots(6,6, figsize=(6,6))
for i in range(36):
    ax.flat[i].imshow(masks_growing_cluster[i])
    ax.flat[i].axis('off')


#%%
aa = torch.tensor(np.float32(aa_x>=i)*0.02).cpu().numpy()
# pl.imshow()
aa

#%%
print(noise.shape, noise.min(), noise.max(), noise[250:270,50:70].unique())
pl.hist((noise>0).cpu().numpy());

#%%
LAB=15
pl.imshow(xx==LAB)
XX = np.where(xx==LAB)
cluster0_coords = np.asarray([np.asarray((i,j)) for i,j in zip(XX[0], XX[1])])
XX = np.where(np.logical_and(xx!=LAB, xx>0))

cluster_others_coords = np.asarray([np.asarray((i,j)) for i,j in zip(XX[0], XX[1])])
print(cluster0_coords.shape, cluster_others_coords.shape)
dists = distance.cdist(cluster0_coords, cluster_others_coords)#.min(axis=1)
dists.shape

#%%
clus_number, clus_coord = find_closest_cluster(xx, LAB)
clus_number, clus_coord
#%%

#%%
# xx[np.where(xx == clus_number)]=LAB
# pl.imshow(xx)
clus_number, clus_coord = find_closest_cluster(xx, LAB)
xx[np.where(xx == clus_number)]=LAB
pl.figure()
pl.imshow(xx)

#%%
dist_small = np.where(dists==np.min(dists.min(axis=0)))[1][0]
cluster_others_coords[dist_small]


#%%
XX = np.where(labelled>0)
X = []
Y = []
for i,j in zip(XX[0], XX[1]):
    Y.append(labelled[i,j])
    X.append(np.asarray((i,j)))
X = np.asarray(X)
Y = np.asarray(Y)
X.shape, Y.shape, np.unique(Y)

#%%
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(linkage='complete', n_clusters=10)
clustering.fit(X)

canvas = np.zeros_like(labelled)
labels_ = clustering.labels_+1
for idx, (i,j) in enumerate(zip(XX[0], XX[1])):
    canvas[i,j] = labels_[idx]
pl.imshow(canvas)

#%%


#%% EXTRA
import scipy
from lib.utils_monai import (load_COVID19_v2,
                            load_synthetic_lesions,
                            load_scans,
                            )
from lib.utils_lung_segmentation import get_segmented_lungs, get_max_rect_in_polygon
from skimage.morphology import disk, binary_closing

data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train'
SCAN_NAME = 'volume-covid19-A-0014'
SLICE=34
torch.set_default_tensor_type('torch.FloatTensor') 
images, labels, keys, files_scans = load_COVID19_v2(data_folder, SCAN_NAME)
scan, scan_mask = load_scans(files_scans, keys, 1, SCAN_NAME,mode="synthetic")
scan_slice = scan[...,SLICE]
scan_slice_copy = copy(scan_slice)
scan_slice_segm = get_segmented_lungs(scan_slice_copy)
im4 = (scan_slice_segm > 0) & (scan_slice_segm < .3)
selem = disk(1)
im5 = binary_closing(im4, selem)
my_label, num_label = scipy.ndimage.label(im5)
size = np.bincount(my_label.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = my_label == biggest_label
HEIGHT, WIDTH, Y2, X1, X2, hist_max = get_max_rect_in_polygon(clump_mask, value=1)
Y1 = Y2 - HEIGHT
X2 = X1 + WIDTH
rect = patches.Rectangle((X1, Y1), WIDTH, HEIGHT, linewidth=1, edgecolor='r', facecolor='none')
lung_sample = np.clip(scan_slice[Y1:Y2, X1:X2]*3, 0, 1)
fig, ax = pl.subplots(2,3,figsize=(12,4))
ax[0,0].imshow(scan_slice)
ax[0,1].imshow(scan_slice_segm)
ax[1,0].imshow(im4)
ax[1,1].imshow(im5)
ax[1,2].imshow(clump_mask)


# aa = texture_lesion[0,100:150,100:150]
# bb = binary_erosion(aa>.3)
# bb1 = binary_dilation(bb)
# bb1 = distance_transform_bf(bb)
# bb = binary_erosion(aa>.15)
# bb = binary_dilation(bb)
# bb = distance_transform_bf(bb)
# xx = bb+bb1*2
# labelled, nr = label(xx)
# xx = labelled * ((bb>0).astype(int)-(binary_erosion(bb>0)).astype(int))

# fig, ax = pl.subplots(2,3, figsize=(12,8))
# ax[0,0].imshow(aa)
# ax[0,1].imshow(bb1)
# ax[0,2].imshow(bb)
# ax[1,0].imshow(bb+bb1*2)
# ax[1,1].imshow(labelled)
# ax[1,2].imshow(xx)