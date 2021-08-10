# IMPORTS
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

from lib.CAModel import CAModel

from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import logging

from lib.utils_plots import visualize_batch, plot_loss, plot_max_intensity_and_mse, plot_lesion_growing, load_baselines, make_seed_1ch, plot_seeds
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
# from lib.utils_cell_auto import prepare_seed
import monai

# FUNCTIONS

def prepare_seed(target, this_seed, device, num_channels = 16, pool_size = 1024):
    # prepare seed
    height, width, _ = np.shape(target)
    seed = np.zeros([height, width, num_channels], np.float32)
    for i in range(num_channels-1):
        seed[..., i+1] = this_seed
    return seed

def load_1ch_py_array(path):
    path = f'{path}/lesion.npz'
    array = np.load(path)
    array = array.f.arr_0
    array = np.expand_dims(array,-1)
    array = np.repeat(array,2,-1)
    array[...,1] = array[...,0]>0 
    return array

def config_cellular_automata(orig_dir, CHANNEL_N, CELL_FIRE_RATE, device, SCALE_GROWTH, model_path, lr, betas_0, betas_1, lr_gamma):
    # h, w = height_width
    # seed = make_seed_1ch((h, w), CHANNEL_N)
    # pool = SamplePool(x=np.repeat(seed[None, ...], cfg.POOL_SIZE, 0))

    ca = CAModel(CHANNEL_N, CELL_FIRE_RATE, device, scale_growth=SCALE_GROWTH).to(device)
    # ca.load_state_dict(torch.load(f'{orig_dir}/{model_path}'))

    optimizer = optim.Adam(ca.parameters(), lr=lr, betas=(betas_0, betas_1))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

    return ca, optimizer, scheduler#, seed

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


@hydra.main(config_path="config", config_name="config.yaml")
def main_train(cfg: DictConfig):
    log = logging.getLogger(__name__)
    path_orig = hydra.utils.get_original_cwd()
    # LOAD FILES
    images, labels, keys, files_scans = load_COVID19_v2(cfg.data.data_folder, cfg.data.SCAN_NAME)
    name_prefix = load_synthetic_lesions(files_scans, keys, cfg.data.BATCH_SIZE)
    scan, scan_mask = load_scans(files_scans, keys, cfg.data.BATCH_SIZE, cfg.data.SCAN_NAME)
    path_single_lesions = f'{cfg.data.path_single_lesions}{cfg.data.SCAN_NAME}_ct/'
    loader_lesions = load_individual_lesions(path_single_lesions, cfg.data.BATCH_SIZE)
    texture = load_synthetic_texture(cfg.data.path_texture)
    print(scan.shape, scan_mask.shape, texture.shape)
    # SUPERPIXELS
    mask_sizes=[]
    cluster_sizes = []
    targets_all = []
    flag_only_one_slice = False
    for idx_mini_batch,mini_batch in enumerate(loader_lesions):
        if idx_mini_batch < cfg.loop.SKIP_LESIONS:continue #resume incomplete reconstructions

        img = mini_batch['image'].numpy()
        mask = mini_batch['label'].numpy()
        mask = remove_small_objects(mask, 20)
        mask_sizes.append([idx_mini_batch, np.sum(mask)])
        name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.npy')[0].split('19-')[-1]
        img_lesion = img*mask

        # if 2nd argument is provided then only analyze that slice
        if cfg.loop.ONLY_ONE_SLICE != -1: 
            slice_used = int(name_prefix.split('_')[-1])
            if slice_used != int(cfg.loop.ONLY_ONE_SLICE): continue
            else: flag_only_one_slice = True

        mask_slic, boundaries, segments, numSegments = boundaries_superpixels(img[0], mask[0])
        segments_sizes = how_large_is_each_segment(segments)

        print(f'img = {np.shape(img)}')

        tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
        targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=cfg.seed.SEED_VALUE, seed_method='max')
        targets_all.append(len(targets))

        coords_big = [int(i) for i in name_prefix.split('_')[1:]]
        fig_superpixels_only_lesions('./', name_prefix, scan, scan_mask, img, mask_slic, boundaries, segments, segments_sizes, coords_big, cfg.loop.TRESH_PLOT, idx_mini_batch, numSegments)
        if flag_only_one_slice: break
    plot_seeds(targets,seeds)
    
    for i in targets:
        print(i.shape)
    
    # path_data = f'{path_orig}/data/{cfg.LESION}'
    # loss_base, max_base, mse_base, loss_base2, max_base2, mse_base2 =  load_baselines(path_data)
    # target_img = load_1ch_py_array(path_data)
    # CELLULAR AUTOMATA
    for idx_tgt, (target_i, seed_i) in enumerate(zip(targets,seeds)):
        seed = prepare_seed(target_i, seed_i, 'cuda', num_channels = cfg.CHANNEL_N, pool_size = 1024)
        pad_target, height_width = pad_target_func(target_i, cfg.TARGET_PADDING, cfg.device)
        ca, optimizer, scheduler = config_cellular_automata(path_orig, cfg.CHANNEL_N, cfg.CELL_FIRE_RATE, cfg.device, cfg.SCALE_GROWTH,  cfg.model_path, cfg.lr, cfg.betas_0, cfg.betas_1, cfg.lr_gamma)
        extra_text = f'_{cfg.data.SCAN_NAME}_{cfg.data.SLICE}_{idx_tgt}'
        loss_log = []
        for i in tqdm(range(cfg.EPOCHS+1)):
        
            x0 = np.repeat(seed[None, ...], cfg.BATCH_SIZE, 0)
            x0 = torch.from_numpy(x0.astype(np.float32)).to(cfg.device)

            x, loss = train(ca, x0, pad_target, np.random.randint(64,96), optimizer, scheduler)
            loss_log.append(loss.item())
            log.info(f"loss = {loss.item()}")

        grow = torch.tensor(seed).unsqueeze(0).to(cfg.device)
        grow_sel = []
        grow_max = []
        mse_recons = []
        target_padded = pad_target.detach().cpu().numpy()[0,...,0]
        with torch.no_grad():
            for i in range(cfg.ITER_GROW):
                grow = ca(grow)
                if i % (cfg.ITER_GROW // cfg.ITER_SAVE) == 0:
                    grow_img = grow.detach().cpu().numpy()
                    grow_img = np.squeeze(np.clip(grow_img[0,...,:1],0,1))
                    mse_recons.append(np.mean((grow_img - target_padded)**2))
                    grow_sel.append(grow_img)
                    grow_max.append(np.max(grow_img))
        
        max_2k, mse_2k, train_loss_2k, max_10k, mse_10k, train_loss_10k = load_baselines(path_orig, extra_text)
        visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy(), text=extra_text)
        plot_loss(loss_log, cfg.SCALE_GROWTH, train_loss_2k, text=extra_text)#, loss_base)
        plot_max_intensity_and_mse(grow_max, mse_recons, cfg.SCALE_GROWTH, max_base=max_2k, max_base2=max_10k, mse_base=mse_2k, mse_base2=mse_10k, text=extra_text)
        plot_lesion_growing(grow_sel, target_i, cfg.ITER_SAVE, text=f'_{cfg.data.SCAN_NAME}_{cfg.data.SLICE}')
        print(pad_target.shape, height_width)

        if cfg.SAVE_NUMPY:
            np.save(f'train_loss_SG=1_ep={cfg.EPOCHS//1000}k{extra_text}.npy', loss_log)
            np.save(f'max_syn_SG=1_ep={cfg.EPOCHS//1000}k{extra_text}.npy', grow_max)
            np.save(f'mse_syn_SG=1_ep={cfg.EPOCHS//1000}k{extra_text}.npy', mse_recons)

if __name__ == "__main__":
    main_train()