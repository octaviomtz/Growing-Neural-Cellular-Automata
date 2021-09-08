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

from lib.utils_monai import (load_COVID19_v2,
                            load_synthetic_lesions,
                            load_scans,
                            load_individual_lesions,
                            load_synthetic_texture)
from lib.utils_lung_segmentation import get_segmented_lungs, get_max_rect_in_polygon
from lib.utils_superpixels import coords_min_max_2D
import scipy
# %%
device='cuda'
vgg16 = models.vgg16(pretrained=True).features

def calc_styles(imgs):
    style_layers = [1, 6, 11, 18, 25]  
    mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
    std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]
    # print(mean.is_cuda, std.is_cuda, imgs.is_cuda)
    x = (imgs-mean) / std
    grams = []
    for i, layer in enumerate(vgg16[:max(style_layers)+1]):
        x = layer(x)
        if i in style_layers:
            h, w = x.shape[-2:]
            y = x.clone()  # workaround for pytorch in-place modification bug(?)
            gram = torch.einsum('bchw, bdhw -> bcd', y, y) / (h*w)
            grams.append(gram)
    return grams

def style_loss(grams_x, grams_y):
    loss = 0.0
    for x, y in zip(grams_x, grams_y):
        loss = loss + (x-y).square().mean()
    return loss

def to_nchw(img):
    img = torch.as_tensor(img)
    if len(img.shape) == 3:
        img = img[None,...]
    return img.permute(0, 3, 1, 2)
# %% Minimalistic Neural CA
ident = torch.tensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
sobel_x = torch.tensor([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])/8.0
lap = torch.tensor([[1.0,2.0,1.0],[2.0,-12,2.0],[1.0,2.0,1.0]])/16.0

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

  def forward(self, x, update_rate=0.5):
    y = perception(x)
    y = self.w2(torch.relu(self.w1(y)))
    b, c, h, w = y.shape
    udpate_mask = (torch.rand(b, 1, h, w)+update_rate).floor()
    return x+y*udpate_mask

  def seed(self, n, sz=128):
    return torch.zeros(n, self.chn, sz, sz)

def to_rgb(x):
  return x[...,:3,:,:]+0.5


# %%
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train'
SCAN_NAME = 'volume-covid19-A-0014'
SLICE=34
torch.set_default_tensor_type('torch.FloatTensor') 
images, labels, keys, files_scans = load_COVID19_v2(data_folder, SCAN_NAME)
scan, scan_mask = load_scans(files_scans, keys, 1, SCAN_NAME,mode="synthetic")
scan_slice = scan[...,SLICE]
scan_slice_copy = copy(scan_slice)
scan_slice_segm = get_segmented_lungs(scan_slice_copy)
print(np.shape(scan_slice), np.shape(scan_mask))
fig, ax = pl.subplots(1,2,figsize=(12,4))
ax[0].imshow(scan_slice)
ax[1].imshow(scan_slice_segm)

# %% GET LARGEST LUNG
from skimage.morphology import disk, binary_closing
im4 = (scan_slice_segm > 0) & (scan_slice_segm < .3)
selem = disk(1)
im5 = binary_closing(im4, selem)
my_label, num_label = scipy.ndimage.label(im5)
size = np.bincount(my_label.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = my_label == biggest_label
fig, ax = pl.subplots(1,3,figsize=(12,4))
ax[0].imshow(im4)
ax[1].imshow(im5)
ax[2].imshow(clump_mask)

# %%
HEIGHT, WIDTH, Y2, X1, X2, hist_max = get_max_rect_in_polygon(clump_mask, value=1)
Y1 = Y2 - HEIGHT
X2 = X1 + WIDTH
rect = patches.Rectangle((X1, Y1), WIDTH, HEIGHT, linewidth=1, edgecolor='r', facecolor='none')
lung_sample = np.clip(scan_slice[Y1:Y2, X1:X2]*3, 0, 1)
fig, ax = pl.subplots(1,3, figsize=(12,4))
ax[0].imshow(clump_mask)
ax[0].add_patch(rect)
ax[1].imshow(lung_sample)
ax[2].hist(lung_sample.flatten())
HEIGHT, WIDTH, Y2, X1, X2

#%% CONVERT TO pseudo RGB
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# lung_sample.to(device)
style_img = np.expand_dims(lung_sample,-1)
style_img = np.repeat(style_img,3,-1)
pl.imshow(style_img)
style_img = np.expand_dims(style_img,0)
style_img = torch.from_numpy(np.float32(style_img)).to(device)
with torch.no_grad():
  target_style = calc_styles(to_nchw(style_img))
for i in (target_style):
    print(np.shape(i))

#%% TEMP
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# style_url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/bubbly/bubbly_0101.jpg'
# style_img = imread(style_url, max_size=128)
# print(type(style_img))
# pl.imshow(style_img)
# style_img = torch.from_numpy(np.float32(style_img)).to(device)
# target_style = calc_styles(to_nchw(style_img))


#%% setup training
ca = CA() 
opt = torch.optim.Adam(ca.parameters(), 1e-3)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], 0.3)
loss_log = []
with torch.no_grad():
  pool = ca.seed(1024)


#%% training loop 
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
for i in range(600):
  with torch.no_grad():
    batch_idx = np.random.choice(len(pool), 4, replace=False)
    pool[batch_idx]
    x = pool[batch_idx]
    if i%2 == 0:   # every second batch contains the seed
      x[:1] = ca.seed(1)
  step_n = np.random.randint(32, 96)
  for k in range(step_n):
    x = ca(x)
  imgs = to_rgb(x)
  styles = calc_styles(imgs)
  loss = style_loss(styles, target_style)
  with torch.no_grad():
    loss.backward()
    for p in ca.parameters():
      p.grad /= (p.grad.norm()+1e-8)   # normalize gradients 
    opt.step()
    opt.zero_grad()
    lr_sched.step()
    batch_idx = torch.from_numpy(batch_idx)
    pool[batch_idx]
    # print(f'type = {type(pool), pool[batch_idx].dtype, x.dtype, pool.dtype}')
    # print(pool[batch_idx].is_cuda, x.is_cuda)
    pool[batch_idx] = x                # update pool
    
    loss_log.append(loss.item())
    if i%200==0:
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
        ' lr:', lr_sched.get_lr()[0], end='')

#%%
# https://stackoverflow.com/questions/2478447/find-largest-rectangle-containing-only-zeros-in-an-n%C3%97n-binary-matrix