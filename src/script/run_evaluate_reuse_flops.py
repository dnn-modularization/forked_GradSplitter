import os
import sys

sys.path.append('..')
from utils.configure_loader import load_configure
from utils.model_tools import powerset

# model = 'simcnn'
# model = 'resnet18'
model = 'mobilenet'

# dataset = 'svhn'
# dataset = 'cifar10'
dataset = 'cifar100'

configs = load_configure(model, dataset)
configs.set_estimator_idx()
estimator_idx = configs.estimator_idx

# composition_task_file_name = f"target_classes.num_classes_{configs.num_classes}.list"
composition_task_file_name = f"target_classes.num_classes_{configs.num_classes}.rep_tasks.list"

composition_task_file_path = os.path.join(configs.data_dir, composition_task_file_name)

cmd = f'python -u ../experiments/compare_flops.py ' \
f'--model {model} --dataset {dataset} --estimator_idx {estimator_idx} --composition_task_file_path {composition_task_file_path} ' \
f'> module_evaluation/modularization_eval.{model}_{dataset}.flops.log'
print(cmd)
os.system(cmd)

