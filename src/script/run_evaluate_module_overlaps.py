import os
import sys

sys.path.append('..')
from utils.configure_loader import load_configure
from utils.model_tools import powerset

model = 'simcnn'
# model = 'resnet18'
# model = 'mobilenet'

# dataset = 'svhn'
# dataset = 'cifar10'
dataset = 'cifar100'

configs = load_configure(model, dataset)
configs.set_estimator_idx()
estimator_idx = configs.estimator_idx

cmd = f'python -u ../experiments/evaluate_weight_overlaps.py ' \
f'--model {model} --dataset {dataset} --estimator_idx {estimator_idx}' \
f'> module_evaluation/modularization_eval.{model}_{dataset}.overlaps.log'
print(cmd)
os.system(cmd)

