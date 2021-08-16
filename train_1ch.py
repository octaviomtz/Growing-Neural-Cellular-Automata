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
from omegaconf import DictConfig, OmegaConf
import logging

from lib.utils_plots import (visualize_batch,
                            plot_loss_max_intensity_and_mse,
                            plot_lesion_growing,
                            load_baselines,
                            make_seed_1ch,
                            plot_seeds,
                            save_cell_auto_reconstruction_vars)
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
import wandb

# FUNCTIONS
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

def train(ca, x, target, steps, optimizer, scheduler, cfg_wandb):
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
    # if cfg_wandb: wandb.log({"train_loss": loss})
    return x, loss


@hydra.main(config_path="config", config_name="config.yaml")
def main_train(cfg: DictConfig):
    # WEIGHTS AND BIASES
    hyperparams_default = {'SCALE_GROWTH':cfg.SCALE_GROWTH, 'SCALE_GROWTH_SYN':cfg.SCALE_GROWTH_SYN,'lr':cfg.lr,'wandb':True}
    if cfg.wandb.save:
        wandb.init(project=cfg.wandb.name, entity='octaviomtz', config=hyperparams_default)
        config = wandb.config
        wandb_omega_config = OmegaConf.create(wandb.config._as_dict())
        cfg.SCALE_GROWTH = wandb_omega_config.SCALE_GROWTH
        cfg.SCALE_GROWTH_SYN = wandb_omega_config.SCALE_GROWTH_SYN
        cfg.lr = wandb_omega_config.lr
        
    # HYDRA
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
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
    flag_slice_found = False
    for idx_mini_batch,mini_batch in enumerate(loader_lesions):
        if idx_mini_batch < cfg.loop.SKIP_LESIONS:continue #resume incomplete reconstructions

        img = mini_batch['image'].numpy()
        mask = mini_batch['label'].numpy()
        mask = remove_small_objects(mask, 20)
        mask_sizes.append([idx_mini_batch, np.sum(mask)])
        name_prefix = mini_batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.npy')[0].split('19-')[-1]
        img_lesion = img*mask

        # if 2nd argument is provided then only analyze that slice
        if cfg.loop.ONLY_ONE_SLICE: 
            slice_used = int(name_prefix.split('_')[-1])
            if slice_used != int(cfg.data.SLICE): continue
            else: flag_slice_found = True

        mask_slic, boundaries, segments, numSegments = boundaries_superpixels(img[0], mask[0])
        segments_sizes = how_large_is_each_segment(segments)

        print(f'img = {np.shape(img)}')

        tgt_minis, tgt_minis_coords, tgt_minis_masks, tgt_minis_big, tgt_minis_coords_big, tgt_minis_masks_big = select_lesions_match_conditions2(segments, img[0], skip_index=0)
        targets, coords, masks, seeds = make_list_of_targets_and_seeds(tgt_minis, tgt_minis_coords, tgt_minis_masks, seed_value=cfg.seed.SEED_VALUE, seed_method='max')
        targets_all.append(len(targets))

        coords_big = [int(i) for i in name_prefix.split('_')[1:]]
        fig_superpixels_only_lesions('./', name_prefix, scan, scan_mask, img, mask_slic, boundaries, segments, segments_sizes, coords_big, cfg.loop.TRESH_PLOT, idx_mini_batch, numSegments)
        if flag_slice_found: break
    
    if flag_slice_found:
        plot_seeds(targets,seeds)
        for i in targets:
            print(i.shape)
    
    # CELLULAR AUTOMATA
    if flag_slice_found:
        for idx_lesion, (target, coord, mask, this_seed) in enumerate(zip(targets, coords, masks, seeds)):
            if cfg.loop.ONLY_ONE_LESION != False and cfg.loop.ONLY_ONE_LESION != idx_lesion: continue
            seed = prepare_seed(target, this_seed, 'cuda', num_channels = cfg.CHANNEL_N, pool_size = 1024)
            pad_target, height_width = pad_target_func(target, cfg.TARGET_PADDING, cfg.device)
            ca, optimizer, scheduler = config_cellular_automata(path_orig, cfg.CHANNEL_N, cfg.CELL_FIRE_RATE, cfg.device, cfg.SCALE_GROWTH,  cfg.model_path, cfg.lr, cfg.betas_0, cfg.betas_1, cfg.lr_gamma)
            if cfg.wandb.save: wandb.watch(ca)
            extra_text = f'_{cfg.data.SCAN_NAME}_{cfg.data.SLICE}_{idx_lesion}'
            loss_log = []
            for i in tqdm(range(cfg.EPOCHS+1)):
            
                x0 = np.repeat(seed[None, ...], cfg.BATCH_SIZE, 0)
                x0 = torch.from_numpy(x0.astype(np.float32)).to(cfg.device)

                x, loss = train(ca, x0, pad_target, np.random.randint(64,96), optimizer, scheduler, cfg.wandb.save)
                loss_log.append(loss.item())
                log.info(f"loss = {loss.item()}")
                if cfg.wandb.save: wandb.log({"train_loss":loss.item()})
            # RECONSTRUCTION
            grow = torch.tensor(seed).unsqueeze(0).to(cfg.device)
            grow_sel = []
            grow_max = []
            mse_recons = []
            target_padded = pad_target.detach().cpu().numpy()[0,...,0]
            with torch.no_grad():
                for i in range(cfg.ITER_GROW):
                    grow = ca(grow, scale_growth_synthesis=cfg.SCALE_GROWTH_SYN)
                    if i % (cfg.ITER_GROW // cfg.ITER_SAVE) == 0:
                        grow_img = grow.detach().cpu().numpy()
                        grow_img = np.squeeze(np.clip(grow_img[0,...,:1],0,1))
                        grow_img = grow_img * mask
                        mse_recons_item = np.mean((grow_img - target_padded)**2)
                        mse_recons.append(mse_recons_item)
                        grow_sel.append(grow_img)
                        grow_max.append(np.max(grow_img))
                        if cfg.wandb.save: 
                            wandb.log({"mse_recons": mse_recons_item})
                            wandb.log({"grow_max": np.max(grow_img)})
            if cfg.wandb.save: wandb.log({"intense_mse": (10000*loss.item())+(10000*mse_recons_item)+np.max(grow_img) })
            print(f'grow_img={grow_img.shape}, mask={mask.shape}')
            max_2k, mse_2k, train_loss_2k, max_10k, mse_10k, train_loss_10k = load_baselines(path_orig, extra_text)
            visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy(), text=extra_text)
            plot_loss_max_intensity_and_mse(loss_log, train_loss_2k, cfg.SCALE_GROWTH, cfg.SCALE_GROWTH_SYN, grow_max, mse_recons,max_base=max_2k, max_base2=max_10k, mse_base=mse_2k, mse_base2=mse_10k, save_wandb=cfg.wandb.save, text=extra_text)
            plot_lesion_growing(grow_sel, target, cfg.ITER_SAVE, text=f'_{cfg.data.SCAN_NAME}_{cfg.data.SLICE}')
            if cfg.save_reconstruction: save_cell_auto_reconstruction_vars(grow_sel, coord, mask, loss_log, name_prefix, idx_lesion)
            print('SYNTHESIS COMPLETED', pad_target.shape, height_width)

            if cfg.SAVE_NUMPY:
                np.save(f'train_loss_SG=1_ep={cfg.EPOCHS//1000}k{extra_text}.npy', loss_log)
                np.save(f'max_syn_SG=1_ep={cfg.EPOCHS//1000}k{extra_text}.npy', grow_max)
                np.save(f'mse_syn_SG=1_ep={cfg.EPOCHS//1000}k{extra_text}.npy', mse_recons)
    else:
        print('SLICE HAS NO LESIONS')




if __name__ == "__main__":
    main_train()