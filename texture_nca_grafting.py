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
style_urls = {
  'dots': 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/dotted/dotted_0090.jpg',
  'chess': 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/chequered/chequered_0121.jpg',
  'bubbles': 'https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/bubbly/bubbly_0101.jpg',
}

imgs = [imread(url, max_size=128) for url in style_urls.values()]
imshow(np.hstack(imgs))
target_name = 'bubbles'
style_img = imread(style_urls[target_name], max_size=128)
with torch.no_grad():
  target_style = calc_styles(to_nchw(style_img))
imshow(style_img)
style_img.shape, type(style_img)

#%%
# print(style_imgs[0].shape)
texture_lungs = np.load('data/texture_lung_inpainted.npy')
texture_lesion = np.load('data/texture_lesion_high2.5_inpain.npy')
texture_lesion = np.repeat(texture_lesion[None,...],3,0)
print(texture_lungs.shape, texture_lesion.shape)
texture_lungs = np.moveaxis(texture_lungs,0,-1) 
texture_lesion = np.moveaxis(texture_lesion,0,-1)
texture_lungs = texture_lungs[:128,:128,:]
texture_lesion = texture_lesion[:128,:128,:]
style_img =  texture_lesion  #  texture_lungs
with torch.no_grad():
    target_style = calc_styles(to_nchw(style_img.astype('float32')))
imshow(style_img)

#%%
style_img.dtype

#%%Loading pretrained models
# !wget -nc https://github.com/google-research/self-organising-systems/raw/master/assets/grafting_nca.zip && unzip grafting_nca.zip

# %% FINETUNNING
#@title setup training
target_name = 'covid'
parent = 'models/init_lungs' # 'dots' replace this with 'init' to train from scratch
model_name = parent+'_'+target_name
ca = torch.load(parent+'.pt')
opt = torch.optim.Adam(ca.parameters(), 1e-3)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], 0.3)
loss_log = []
with torch.no_grad():
  pool = ca.seed(256)

# %% TRAINING
#@title training loop {vertical-output: true}
for i in range(1000):
  with torch.no_grad():
    batch_idx = np.random.choice(len(pool), 4, replace=False)
    x = pool[batch_idx]
    if i%8 == 0:
      x[:1] = ca.seed(1)
  step_n = np.random.randint(32, 96)
  for k in range(step_n):
    x = ca(x)
  imgs = to_rgb(x)
  styles = calc_styles(imgs)
  overflow_loss = (x-x.clamp(-1.0, 1.0)).abs().sum()
  loss = style_loss(styles, target_style)+overflow_loss
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
      torch.save(ca, model_name+'.pt')
    if i%10 == 0:
      print('\rstep_n:', len(loss_log),
        ' loss:', loss.item(), 
        ' lr:', lr_sched.get_lr()[0], end='')

#%%
#@title NCA video {vertical-output: true}
def show_ca(ca, path):
  with VideoWriter(path) as vid, torch.no_grad():
    x = ca.seed(1, 256)
    for k in tnrange(300, leave=False):
      step_n = min(2**(k//30), 16)
      for i in range(step_n):
        x[:] = ca(x)
      img = to_rgb(x[0]).permute(1, 2, 0).cpu()
      vid.add(zoom(img, 2))

show_ca(torch.load('models/init_lungs.pt'), './temp_vids/init_lungs.mp4')
show_ca(torch.load('models/init_lungs_covid.pt'), './temp_vids/init_lungs_covid.mp4')

#%%
print(x.shape)
pl.imshow(x[0,0,...].detach().cpu().numpy())

# %%
W = 256
with torch.no_grad():
  r = torch.linspace(-1, 1, W)**2
  r = (r+r[:,None]).sqrt()
  mask = ((0.006-r)*8.0).sigmoid() #.6, .8
  mask = (mask - torch.min(mask)) / (torch.max(mask) - torch.min(mask))
  # pl.contourf(mask.cpu())
  # pl.colorbar()
  # pl.axis('equal')

pl.imshow(mask.detach().cpu().numpy())
pl.colorbar()
pl.axis('equal')

# %%
ca1 = torch.load('models/init_lungs.pt')
ca2 = torch.load('models/init_lungs_covid.pt')
with VideoWriter('./temp_vids/ca1_ca2.mp4') as vid, torch.no_grad():
  x = torch.zeros([1, ca1.chn, W, W])
  for i in tnrange(600):
    for k in range(8):
      x1, x2 = ca1(x), ca2(x)
      x = x1 + (x2-x1)*mask
    img = to_rgb(x[0]).permute(1, 2, 0).cpu()
    vid.add(zoom(img, 2))

#%%
def growing_circle(iter):
    radious = iter / 10000
    W = 256
    with torch.no_grad():
        r = torch.linspace(-1, 1, W)**2
        r = (r+r[:,None]).sqrt()
        mask = ((radious - r) * 8.0).sigmoid() #.6, .8
        mask = (mask - torch.min(mask)) / (torch.max(mask) - torch.min(mask))
    return mask

ca1 = torch.load('models/init_lungs.pt')
ca2 = torch.load('models/init_lungs_covid.pt')
with VideoWriter('./temp_vids/growing_circleC.mp4') as vid, torch.no_grad():
  x = torch.zeros([1, ca1.chn, W, W])
  for i in tnrange(600): 
    x[:] = ca1(x)
  x1 = x
  x = torch.zeros([1, ca1.chn, W, W])
  for i in tnrange(600):
    mask = growing_circle(i)
    for i in tnrange(60):
        x[:] = ca2(x)
    x2 = x
    x = x1 + (x2-x1)*mask
    img = to_rgb(x[0]).permute(1, 2, 0).cpu()
    vid.add(zoom(img, 2))

#%%
def growing_circle2(iter):
    radious = iter
    W = 256
    with torch.no_grad():
        r = torch.linspace(-1, 1, W)**2
        r = (r+r[:,None]).sqrt()
        mask = 1-10/((100+(1*torch.exp(1-r)))*1) #.6, .8
        mask = (mask - torch.min(mask)) / (torch.max(mask) - torch.min(mask))
    return mask
a = growing_circle2(.1).detach().cpu().numpy()
pl.imshow(a)
pl.colorbar()
pl.axis('equal')

#%%

# %%
def show_pair(fn1, fn2, W=512):
  ca1 = torch.load(fn1)
  ca2 = torch.load(fn2)
  with VideoWriter('./temp_vids/show_pair.mp4') as vid, torch.no_grad():
    x = torch.zeros([1, ca1.chn, 128, W])
    mask = 0.5-0.5*torch.linspace(0, 2.0*np.pi, W).cos()
    for i in tnrange(300, leave=False):
      for k in range(8):
        x1, x2 = ca1(x), ca2(x)
        x = x1 + (x2-x1)*mask
      img = to_rgb(x[0]).permute(1, 2, 0).cpu()
      vid.add(zoom(img, 2))

# show_pair('chess.pt', 'dots_bubbles.pt')
show_pair('dots_chess.pt', 'dots_bubbles.pt')
# %%
pl.imshow(mask.cpu())
pl.colorbar()
# %%
