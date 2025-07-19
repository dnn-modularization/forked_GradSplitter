import random
import numpy as np
import torch

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# slow & deterministic
torch.backends.cudnn.deterministic = True

class GlobalConfigures:
    def __init__(self):
        import os
        root_dir = f'{}/forked_GradSplitter'  # modify the dir
        assert os.path.exists(root_dir)

        if os.path.exists(f'{root_dir}/data'):
            self.data_dir = f'{root_dir}/data'
        else:
            raise ValueError(f'{root_dir}/data does not exist.')
        self.dataset_dir = f'{self.data_dir}/dataset'
        self.trained_entire_model_name = 'entire_model.pth'
        self.estimator_idx = seed 

        # Set in directory configures/
        self.workspace = None
        self.best_batch_size = None
        self.best_alpha = None
        self.best_lr = None
        self.best_epoch = None

        # define after setting estimator_idx
        self.trained_model_dir = None
        self.trained_model_path = None
        self.module_save_dir = None
        self.best_module_path = None

    def set_estimator_idx(self, idx=seed):
        self.estimator_idx = idx
        self.trained_model_dir = f'{self.workspace}/trained_models'
        self.trained_model_path = f'{self.trained_model_dir}/estimator_{self.estimator_idx}.pth'
        self.module_save_dir = f'{self.workspace}/modules/estimator_{self.estimator_idx}'
        self.best_module_path = f'{self.module_save_dir}/estimator_{self.estimator_idx}.pth'

