#%% TO DO
# remove the last lines if too many zeros
# inpaint the lines defined by rects
# pass through nca


#%%
import scipy
import torch
import numpy as np
from copy import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from lib.utils_monai import (load_COVID19_v2,
                            load_synthetic_lesions,
                            load_scans,
                            )
from lib.utils_lung_segmentation import get_segmented_lungs, get_max_rect_in_mask, get_roi_from_each_lung, make_mosaic_of_rects
from skimage.morphology import disk, binary_closing
from rectpack import newPacker
from rectpack.guillotine import GuillotineBssfSas
from scipy.ndimage import label

#%%
data_folder = '/content/drive/MyDrive/Datasets/covid19/COVID-19-20_v2/Train'
SCAN_NAME = 'volume-covid19-A-0014'
SLICE=34
torch.set_default_tensor_type('torch.FloatTensor') 
images, labels, keys, files_scans = load_COVID19_v2(data_folder, SCAN_NAME)
scan, scan_mask = load_scans(files_scans, keys, 1, SCAN_NAME,mode="synthetic")

lung_samples=[]
lesion_samples=[]
for SLICE in np.arange(2,scan.shape[-1]-2):
    scan_slice = scan[...,SLICE]
    scan_slice_mask = scan_mask[...,SLICE]
    scan_slice_copy = copy(scan_slice)
    scan_slice_segm = get_segmented_lungs(scan_slice_copy)
    lung0, lung1 = get_roi_from_each_lung(scan_slice_segm)
    # remove the lesion if its there
    lung0, lung1 = lung0*np.abs(1-scan_slice_mask), lung1*np.abs(1-scan_slice_mask)
    if len(np.unique(lung0)) < 2: continue
    Y1, X1, Y2, X2 = get_max_rect_in_mask(lung0)
    lung_samples.append(scan_slice[Y1:Y2, X1:X2])
    if len(np.unique(lung1)) < 2: continue
    Y1, X1, Y2, X2 = get_max_rect_in_mask(lung1)
    lung_samples.append(scan_slice[Y1:Y2, X1:X2])
    # LESIONS
    if np.sum(scan_slice_mask) > 5:
        Y1, X1, Y2, X2 = get_max_rect_in_mask(scan_slice_mask)
        lesion_samples.append(scan_slice[Y1:Y2, X1:X2])
print('done')

#%%
sum_areas=0
for i in lesion_samples:
    sum_areas += i.shape[0]*i.shape[1]
sum_areas ** (1/2)

#%% Put all samples in a bin
bin0=(350,350)
lung_samples2 = [i for i in lung_samples if 0 not in i.shape]
print(len(lung_samples2))
rects_lung = [i.shape for i in lung_samples2]
bins = [bin0, (80, 40), (200, 150)]
packer = newPacker()
for r_idx, r in enumerate(rects_lung):
	packer.add_rect(r[0],r[1],r_idx)
for b in bins:
	packer.add_bin(*b)
packer.pack()
print(len(packer))
all_rects = packer.rect_list()

mosaic, other_vars = make_mosaic_of_rects(all_rects, lung_samples2, bin0)

#%%
fig, ax = plt.subplots(1, figsize=(8,8))
ax.imshow(mosaic)
for rect in packer[0]:
    rect_patches = patches.Rectangle((rect.x, rect.y), rect.width, rect.height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect_patches)

# %%
for i,j,k, l, m in zip(idx_error, conds, shape_lung, shape_rec, coords_error):
    print(i,j,k,l,m)


# %% LESIONS
lesion_samples2 = [i for i in lesion_samples if 0 not in i.shape]

if len(lesion_samples2) < 30:
    lesion_samples2 = lesion_samples2 + copy(lesion_samples2) + lesions_small*5
len(lesion_samples2)

bin_les0=(80,80)
rects_lung = [i.shape for i in lesion_samples2]
print(len(lesion_samples2))
bins = [bin_les0, (80, 40), (200, 150)]
packer_les = newPacker()
for r_idx, r in enumerate(rects_lung):
	packer_les.add_rect(r[0],r[1],r_idx)
for b in bins:
	packer_les.add_bin(*b)
packer_les.pack()#
print(len(packer_les))
all_rects = packer_les.rect_list()
#===============
mosaic, other_vars = make_mosaic_of_rects(all_rects, lesion_samples2, bin_les0)
# =================
fig, ax = plt.subplots(1, figsize=(8,8))
ax.imshow(mosaic, vmax=1)
for rect in packer_les[0]:
    rect_patches = patches.Rectangle((rect.x, rect.y), rect.width, rect.height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect_patches)

#%%
np.sum(mosaic==0) / (np.shape(mosaic)[0]*np.shape(mosaic)[1])

# %%
mask_inpain_mosaic = (mosaic==0)
plt.imshow(mask_inpain_mosaic)
# %%
from skimage.restoration import inpaint
image_result = inpaint.inpaint_biharmonic(mosaic, mask_inpain_mosaic, multichannel=False)
plt.imshow(image_result)
# %%
plt.figure(figsize=(8,8))
plt.imshow(mosaic)
# %%
np.sum(mosaic==0, axis=1)

# %%
lesions_small = []
for rect in packer_les[0]:
    if rect.width <= 5 and rect.height <= 5:
        lesions_small.append(lesion_samples2[rect.rid])

# %%
plt.imshow(lesion_samples2[17])
# %%
rect.rid
# %%
