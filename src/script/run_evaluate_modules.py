import os
import sys

sys.path.append('..')
from utils.configure_loader import load_configure

# parallel
model = 'simcnn'
# model = 'resnet18'
# model = 'mobilenet'
dataset = 'cifar10'

configs = load_configure(model, dataset)
configs.set_estimator_idx()
estimator_idx = configs.estimator_idx

cmd = f'python -u ../experiments/evaluate_modules.py ' \
      f'--model {model} --dataset {dataset} --estimator_idx {estimator_idx} ' \
      f'> logs/eval_{model}_{dataset}_estimator_{estimator_idx}.log'
print(cmd)
os.system(cmd)


