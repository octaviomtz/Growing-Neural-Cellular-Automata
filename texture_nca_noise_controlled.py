#%%
import os
import io
from texture_nca import style_loss
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
# %%
#@title setup training
ca = CA() 
opt = torch.optim.Adam(ca.parameters(), 1e-3)
lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000], 0.3)
loss_log = []
with torch.no_grad():
  pool = ca.seed(1024)
# %%
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

# %%
