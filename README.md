# Neural Cellular Automata (nCA) experiments
Experimenting with different parameters of the Cellular Automata on 1-channel images.

Based on the pytorch implementation (github.com/chenmingxiang110/Growing-Neural-Cellular-Automata) of the nCA developed by _**Mordvintsev A.**, et al., "Growing Neural Cellular Automata", Distill, 2020._

## Single Run
We use Hydra to handle different configurations:
1. We save the default parameters in _config/config.yaml_ 
    ```yaml
    # config/config.yaml
    CHANNEL_N: 16
    CELL_FIRE_RATE: 0.5
    lr: 2e-3
    data:
        SCAN_NAME: volume-covid19-A-0014
    loop:
        ONLY_ONE_LESION: 1
    seed:
        SEED_VALUE: 0.19
    ```
    We can run single experiments and override hyperparameters:
    ```bash
    python3 train_1ch.py cfg.loop.ONLY_ONE_LESION=False cfg.seed.SEED_VALUE=1
    ```
1. Hydra saves the results (images, logs, arrays, etc) in individual folders:
    ![alt text](https://github.com/octaviomtz/Growing-Neural-Cellular-Automata/images_github/hydra_folders.png)

## Hyperparameter optimization
'''
wandb sweep sweep.yaml
NUM=20
wandb agent octaviomtz/Growing-Neural-Cellular-Automata/76fjhha0 --count $NUM

'''

