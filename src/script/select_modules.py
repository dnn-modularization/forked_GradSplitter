import os
import sys
sys.path.append('..')
from utils.configure_loader import load_configure

# model = 'simcnn'
# model = 'resnet18'
model = 'mobilenet'

# dataset = 'svhn'
# dataset = 'cifar10'
dataset = 'cifar100'

lr_head = 0.01
lr_modularity = 0.001
alpha = 0.1  # for the weighted sum of loss1 and loss2
batch_size = 64

configs = load_configure(model, dataset)
configs.set_estimator_idx()
module_save_dir = f'{configs.module_save_dir}/lr_{lr_head}_{lr_modularity}_alpha_{alpha}'

cmd = f'cp {module_save_dir}/last_epoch.pth ' \
      f'{configs.module_save_dir}/estimator_{configs.estimator_idx}.pth'
os.system(cmd)
print(cmd)
