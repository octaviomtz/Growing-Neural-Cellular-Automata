import scipy
from skimage.morphology.binary import binary_erosion
import torch
import numpy as np
from copy import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from lib.utils_monai import (load_COVID19_v2,
                            load_synthetic_lesions,
                            load_scans,
                            )
from lib.utils_lung_segmentation import get_segmented_lungs, get_max_rect_in_mask, get_roi_from_each_lung, make_mosaic_of_rects, convert_from_0ch_to_3ch
from skimage.morphology import disk, binary_closing
from rectpack import newPacker
from rectpack.guillotine import GuillotineBssfSas
from scipy.ndimage import label
from tqdm import tqdm
from copy import copy
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from hydra.utils import get_original_cwd
import warnings
warnings.filterwarnings("ignore", message="tqdm is not installed")

def create_mosaic_mask_for_inpainting(mosaic, packer, name='mosaic_mask'):
    """Create a mask (for optional inpainting) that covers all regions with zeros
    and all borders between rects

    Args:
        mosaic (numpy array): [description]
        packer (rectpack Packer object): [description]

    Returns:
        [type]: [description]
    """
    mosaic_border = np.zeros_like(mosaic)
    for rect in packer[0]:
        try:
            mosaic_border[rect.y:rect.y+rect.height, rect.x] = 1
            mosaic_border[rect.y:rect.y+rect.height, rect.x+rect.width] = 1
            mosaic_border[rect.y, rect.x: rect.x+rect.width] = 1
            mosaic_border[rect.y+rect.height, rect.x:rect.x+rect.width] = 1
        except IndexError: continue
    mosaic_mask = np.clip((mosaic==0) + mosaic_border, 0, 1)
    plt.figure(figsize=(8,8))
    plt.imshow(mosaic_mask)
    plt.savefig(name)
    return mosaic_mask

def put_all_rects_in_bins(lung_samples, bin_size, only_high_intens=False, median_intens=.25):
    """Pack a set of rectangles (lung or lesion textures) into a larger rectangle.
    following example in https://github.com/secnot/rectpack

    Args:
        lung_samples (list): list of numpy arrays of all rectangles found
        bin_size (int): size of both sides of the bin
        only_high_intens (bool, optional): discard all rects with a median
        intensity lower than median_intens (specially for lesions of multiple patients). 
        Defaults to False.

    Returns:
        [type]: [description]
    """
    bin0 = (bin_size, bin_size)
    lung_samples2 = [i for i in lung_samples if 0 not in i.shape]
    
    if only_high_intens:
        lesion_samples_high_intens = []
        for i in lung_samples2:
            if np.median(i)>.25:
                lesion_samples_high_intens.append(i)
        lung_samples2 =lesion_samples_high_intens

    print(f'number of rects = {len(lung_samples2)}')
    rects_lung = [i.shape for i in lung_samples2]
    bins = [bin0, (80, 40), (200, 150)]
    packer = newPacker()
    for r_idx, r in enumerate(rects_lung):
        packer.add_rect(r[0],r[1],r_idx)
    for b in bins:
        packer.add_bin(*b)
    packer.pack()
    print(f'number of bins: {len(packer)}')
    all_rects = packer.rect_list()
    return packer, all_rects, lung_samples2

def save_img_mosaic(mosaic, packer, img_name='mosaic'):
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.imshow(mosaic)
    for rect in packer[0]:
        rect_patches = patches.Rectangle((rect.x, rect.y), rect.width, rect.height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_patches)
    fig.savefig(f'{img_name}')
    

@hydra.main(config_path="config", config_name="config_texture.yaml")
def main(cfg: DictConfig):
    data_folder = cfg.data_folder
    SCAN_NAME = cfg.SCAN_NAME
    torch.set_default_tensor_type('torch.FloatTensor') 
    
    files = os.listdir(data_folder)
    files = [i.split('_ct')[0] for i in files if 'ct.nii' in i]
    files = files[:60]

    if SCAN_NAME in files: 
        files = [SCAN_NAME]
    else:
        files = files

    print(np.shape(files))

    # STEP 1. GET ALL RECTANGLES FROM N SCANS
    lung_samples=[]
    lesion_samples=[]
    for SCAN_NAME in tqdm(files):
        if SCAN_NAME in ['volume-covid19-A-0247', 'volume-covid19-A-0504', 'volume-covid19-A-0112']: continue
        images, labels, keys, files_scans = load_COVID19_v2(data_folder, SCAN_NAME)
        scan, scan_mask = load_scans(files_scans, keys, 1, SCAN_NAME,mode="synthetic")

        for SLICE in tqdm(np.arange(2,scan.shape[-1]-2), leave=False):
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
    print(f'done: lung_samples={len(lung_samples)}, lesion_samples={len(lesion_samples)}')

    # STEP 2. PUT ALL RECTS FROM LUNGS IN A BIN
    packer, all_rects, lung_samples2 = put_all_rects_in_bins(lung_samples, cfg.bin_size)
    mosaic, other_vars = make_mosaic_of_rects(all_rects, lung_samples2, (cfg.bin_size,cfg.bin_size))
    save_img_mosaic(mosaic, packer, img_name='mosaic')
    mosaic_mask = create_mosaic_mask_for_inpainting(mosaic, packer)
    if cfg.save_results_locally:
        np.save(f'{get_original_cwd()}/data/texture_lung_mask_', mosaic_mask)
        mosaic_img = convert_from_0ch_to_3ch(mosaic)
        np.save(f'{get_original_cwd()}/data/texture_lung_', mosaic_img)


    #STEP 3. REPEAT WITH LESIONS
    packer, all_rects, lesion_samples2 = put_all_rects_in_bins(lesion_samples, cfg.bin_size_les, only_high_intens=False)
    mosaic, other_vars = make_mosaic_of_rects(all_rects, lesion_samples2, (cfg.bin_size_les,cfg.bin_size_les))
    save_img_mosaic(mosaic, packer, img_name='mosaic_lesion')
    mosaic_les_mask = create_mosaic_mask_for_inpainting(mosaic, packer, name='mosaic_les_mask')
    if cfg.save_results_locally:
        np.save(f'{get_original_cwd()}/data/texture_lesions_mask_', mosaic_les_mask)
        mosaic_img = convert_from_0ch_to_3ch(mosaic)
        np.save(f'{get_original_cwd()}/data/texture_lesion_', mosaic_img)

if __name__ == '__main__':
    main()