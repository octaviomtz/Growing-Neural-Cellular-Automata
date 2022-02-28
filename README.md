# Neural Cellular Automata (nCA) covid-19 lesion synthesis
1. Split large lesions using superpixels into smaller components
1. For each smaller component set a seed and reconstruct it using CA
![hydra_folders](/images_github/lesion_superpixels.png?raw=true)

## Use:
1. To segment (if needed) and train a nCA on the segmented parts of a lesion run
    ```bash
    python3 train_1ch.py 
    ```
1. To insert the synthetic lesions produced by nCA run
    ```bash
    # make sure path_synthesis points to the folder with the results produced by (1)
    python3 replace_lesion.py 
    ```

## Single Runs
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
    We can run single experiments and override hyperparams:
    ```bash
    python3 train_1ch.py cfg.loop.ONLY_ONE_LESION=False cfg.seed.SEED_VALUE=1
    ```
1. Hydra saves the results (images, logs, arrays, etc) in individual folders:
    ![hydra_folders](/images_github/hydra_folders.png?raw=true)

1. For multiple runs just add commas and the --multirun flag
    ```bash
    python3 train_1ch.py data.SLICE=18,19,20,21,22 wandb.name=covid_A-0014 --multirun
    ```
1. For a dry run make sure the _temp_delete_ folder exists
    ```
    python3 train_1ch.py data.SLICE=34 wandb.save=False hydra.run.dir=temp_delete hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled
    ```

## Hyperparameter optimization

We use WANDB _sweeps_ to handle hyperparameter optimization.
1. We save the hyperparams in _sweep.yaml_
    ```yaml
    # sweep.yaml
    parameters:
    SCALE_GROWTH:
        values: [1, .5, .1, .05, .01]
    lr:
        distribution: log_uniform
        min: -4
        max: -1
    ```
1. We initialize the sweep, then define the number of runs to do and finally we launch the sweep
    ```bash
    wandb sweep sweep.yaml
    NUM=20
    wandb agent username/Growing-Neural-Cellular-Automata/76fjhha0 --count $NUM
    ```
    we can compare the models and identify the best hyperparams in wandb:
    ![sweep_results](/images_github/sweep_results.png?raw=true)
1. Caveat: in order to match hydra and wandb we need to reroute the hyperparams:
    1. get the hydra default hyperparams into wand.config
    1. create a temporary omega config
    1. update the hyperparams used of the running cgf
    ```python
    # train_1ch.py
    hyperparams_default = {'SCALE_GROWTH':cfg.SCALE_GROWTH, 'SCALE_GROWTH_SYN':cfg.SCALE_GROWTH_SYN,'lr':cfg.lr,'wandb':True}
    wandb.init(project='cellaut_grid_search', entity='octaviomtz', config=hyperparams_default)
    config = wandb.config
    wandb_omega_config = OmegaConf.create(wandb.config._as_dict())
    cfg.SCALE_GROWTH = wandb_omega_config.SCALE_GROWTH
    cfg.SCALE_GROWTH_SYN = wandb_omega_config.SCALE_GROWTH_SYN
    cfg.lr = wandb_omega_config.lr
    ```

# Testing

We use tox to unit test specifc parts of the program. 
To run the the tests just run
```bash
tox
```
this will check all the tests in the tests/ folder

For example, lib.utils_lung_segmentation.get_max_rect_in_mask finds the largest rectangle in a binary mask (to later create a mosaic for texture synthesis). Then, if the function is updated, tests/test_mask_rect_in_mask.py will check that the correct rectangle is found by the updated function. 
![unit_test_example](/images_github/tox_test_example.png?raw=true)

## Notes
Experimenting with different parameters of the Cellular Automata on 1-channel images.
Based on the pytorch implementation (github.com/chenmingxiang110/Growing-Neural-Cellular-Automata) of the nCA developed by _**Mordvintsev A.**, et al., "Growing Neural Cellular Automata", Distill, 2020._