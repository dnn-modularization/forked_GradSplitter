import sys
sys.path.append('..')


def load_configure(model_name, dataset_name):
    model_dataset_name = f'{model_name}_{dataset_name}'
    if model_dataset_name == 'simcnn_cifar10':
        from configures.simcnn_cifar10 import Configures
    elif model_dataset_name == 'simcnn_svhn':
        from configures.simcnn_svhn import Configures
    elif model_dataset_name == 'simcnn_cifar100':
        from configures.simcnn_cifar100 import Configures
    elif model_dataset_name == 'resnet18_cifar10':
        from configures.resnet18_cifar10 import Configures
    elif model_dataset_name == 'resnet18_svhn':
        from configures.resnet18_svhn import Configures
    elif model_dataset_name == 'resnet18_cifar100':
        from configures.resnet18_cifar100 import Configures
    elif model_dataset_name == 'mobilenet_cifar10':
        from configures.mobilenet_cifar10 import Configures
    elif model_dataset_name == 'mobilenet_svhn':
        from configures.mobilenet_svhn import Configures
    elif model_dataset_name == 'mobilenet_cifar100':
        from configures.mobilenet_cifar100 import Configures
    else:
        raise ValueError()
    configs = Configures()
    return configs
