import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from global_configure import seed

# the seeds for randomly sampling from the original training dataset based on Dirichlet Distribution.
estimator_indices = [seed]

#model = 'simcnn'
#model = 'resnet18'
model="mobilenet"

# dataset = 'cifar10'
dataset = 'svhn'
# dataset = 'cifar100'

for i, estimator_idx in enumerate(estimator_indices):
    cmd = f'python -u ../grad_splitter.py --model {model} --dataset {dataset} ' \
          f'--estimator_idx {estimator_idx} > logs/{model}_{dataset}_estimator_{estimator_idx}.log'
    print(cmd)
    os.system(cmd)

