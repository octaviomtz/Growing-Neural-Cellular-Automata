import os
import numpy as np
import monai
import math
import torch
import glob
from skimage.morphology import remove_small_holes, remove_small_objects
from monai.transforms import (
    LoadImaged,
    AddChanneld,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandFlipd,
    RandFlipd,
    RandFlipd,
    CastToTyped,
)

def get_xforms_scans_or_synthetic_lesions(mode="scans", keys=("image", "label")):
    """returns a composed transform for scans or synthetic lesions."""
    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
    ]
    dtype = (np.int16, np.uint8)
    if mode == "synthetic":
        xforms.extend([
          ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        ])
        dtype = (np.float32, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

def get_xforms_load(mode="load", keys=("image", "label")):
    """returns a composed transform."""
    xforms = [
        LoadImaged(keys),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "load":
        dtype = (np.float32, np.uint8)
    xforms.extend([CastToTyped(keys, dtype=dtype)])
    return monai.transforms.Compose(xforms)

def load_COVID19_v2(data_folder, SCAN_NAME):
    images= [f'{data_folder}/{SCAN_NAME}_ct.nii.gz']
    labels= [f'{data_folder}/{SCAN_NAME}_seg.nii.gz']
    keys = ("image", "label")
    files_scans = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]
    return images, labels, keys, files_scans


def load_synthetic_lesions(files_scans, keys, batch_size):
    transforms_load = get_xforms_scans_or_synthetic_lesions("synthetic", keys)
    ds_synthetic = monai.data.CacheDataset(data=files_scans, transform=transforms_load)
    loader_synthetic = monai.data.DataLoader(
            ds_synthetic,
            batch_size=batch_size,
            shuffle=False, #should be true for training
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
    for idx_mini_batch, mini_batch in enumerate(loader_synthetic):
        # if idx_mini_batch==6:break #OMM
        BATCH_IDX=0
        scan_synthetic = mini_batch['image'][BATCH_IDX][0,...].numpy()
        scan_mask = mini_batch['label'][BATCH_IDX][0,...].numpy()
        name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('Train/')[-1].split('.nii')[0]
        return name_prefix

def load_scans(files_scans, keys, batch_size, SCAN_NAME):
    transforms_load = get_xforms_scans_or_synthetic_lesions("scans", keys)
    ds_scans = monai.data.CacheDataset(data=files_scans, transform=transforms_load)
    loader_scans = monai.data.DataLoader(
            ds_scans,
            batch_size=batch_size,
            shuffle=False, #should be true for training
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    for idx_mini_batch, mini_batch in enumerate(loader_scans):
        # if idx_mini_batch==1:break #OMM
        BATCH_IDX=0
        scan = mini_batch['image'][BATCH_IDX][0,...]
        scan_mask = mini_batch['label'][BATCH_IDX][0,...]
        scan_name = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii')[0][:-3]
    print(f'working on scan= {scan_name}')
    assert scan_name == SCAN_NAME, 'cannot load that scan'
    scan = scan.numpy()   #ONLY READ ONE SCAN (WITH PREVIOUS BREAK)
    scan_mask = scan_mask.numpy()
    return scan, scan_mask

def load_individual_lesions(folder_source, batch_size):
    # folder_source = f'/content/drive/MyDrive/Datasets/covid19/COVID-19-20/individual_lesions/{SCAN_NAME}_ct/'
    files_scan = sorted(glob.glob(os.path.join(folder_source,"*.npy")))
    files_mask = sorted(glob.glob(os.path.join(folder_source,"*.npz")))
    keys = ("image", "label")
    files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(files_scan, files_mask)]
    print(len(files_scan), len(files_mask), len(files))
    transforms_load = get_xforms_load("load", keys)
    ds_lesions = monai.data.CacheDataset(data=files, transform=transforms_load)
    loader_lesions = monai.data.DataLoader(
            ds_lesions,
            batch_size=batch_size,
            shuffle=False, #should be true for training
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
    return loader_lesions

def load_synthetic_texture(path_synthesis_old):
    texture_orig = np.load(f'{path_synthesis_old}texture.npy.npz')
    texture_orig = texture_orig.f.arr_0
    texture = texture_orig + np.abs(np.min(texture_orig))# + .07
    return texture
