import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from lib.utils_monai import (load_COVID19_v2,
                            load_synthetic_lesions,
                            load_scans,
                            load_individual_lesions,
                            load_synthetic_texture)

from lib.utils_replace_lesions import replace_with_nCA

@hydra.main(config_path="config", config_name="config_replace_lesion.yaml")
def main(cfg: DictConfig):       
    # HYDRA
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    path_orig = hydra.utils.get_original_cwd()
    # LOAD FILES
    images, labels, keys, files_scans = load_COVID19_v2(cfg.data.data_folder, cfg.data.SCAN_NAME)
    name_prefix = load_synthetic_lesions(files_scans, keys, cfg.data.BATCH_SIZE)
    scan, scan_mask = load_scans(files_scans, keys, cfg.data.BATCH_SIZE, cfg.data.SCAN_NAME,mode="synthetic")
    path_single_lesions = f'{cfg.data.path_single_lesions}{cfg.data.SCAN_NAME}_ct/'
    loader_lesions = load_individual_lesions(path_single_lesions, cfg.data.BATCH_SIZE)
    texture = load_synthetic_texture(cfg.data.path_texture)
    print(scan.shape, scan_mask.shape, texture.shape)
        
    path_synthesis = '/content/drive/My Drive/repositories/cellular_automata/Growing-Neural-Cellular-Automata/temp_delete/volume-covid19-A-0014/'

    scan_slice = scan[...,cfg.data.SLICE]
    print(np.shape(scan_slice), np.shape(scan_mask), np.shape(texture))
    print(f'scan_slice={np.min(scan_slice), np.max(scan_slice)}, scan_mask={np.min(scan_mask), np.max(scan_mask)}, texture={np.min(texture), np.max(texture)}')
    arrays_seq, seq = replace_with_nCA(path_synthesis, scan_slice, cfg.data.SCAN_NAME, cfg.data.SLICE, texture)


if __name__ == "__main__":
    main()