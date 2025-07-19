from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CIFAR10, SVHN, CIFAR100

def get_dataset_loader(dataset_name):
    if dataset_name == 'cifar10':
        load_dataset = load_cifar10
    elif dataset_name == 'svhn':
        load_dataset = load_svhn
    elif dataset_name == 'cifar100':
        load_dataset = load_cifar100
    elif dataset_name == 'imagenet':   
        load_dataset = load_imagenet
    else:
        raise ValueError
    return load_dataset


def get_dataset_loader_target_class(dataset_name):
    if dataset_name == 'cifar10':
        load_dataset = load_cifar10_target_class
    elif dataset_name == 'svhn':
        load_dataset = load_svhn_target_class
    elif dataset_name == 'cifar100':
        load_dataset = load_cifar100_target_class
    elif dataset_name == 'imagenet':   
        load_dataset = load_imagenet_target_class
    else:
        raise ValueError
    return load_dataset

def load_cifar100(dataset_dir, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR100(root=dataset_dir, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR100(root=dataset_dir, train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_cifar100_target_class(dataset_dir, batch_size, num_workers, target_classes, transform_label=True):
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR100(root=dataset_dir, train=True, transform=transform, download=True)
    train_targets = np.array(train_dataset.targets)
    idx = np.isin(train_targets, target_classes)
    target_label = train_targets[idx].tolist()
    if transform_label:
        trans_label = [target_classes.index(i) for i in target_label]
        train_dataset.targets = trans_label
    else:
        train_dataset.targets = target_label
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR100(root=dataset_dir, train=False,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_targets = np.array(test_dataset.targets)
    idx = np.isin(test_targets, target_classes)
    target_label = test_targets[idx].tolist()
    if transform_label:
        trans_label = [target_classes.index(i) for i in target_label]
        test_dataset.targets = trans_label
    else:
        test_dataset.targets = target_label
    test_dataset.data = test_dataset.data[idx]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_cifar10(dataset_dir, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR10(root=dataset_dir, train=False, download=True,
                           transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_cifar10_target_class(dataset_dir, batch_size, num_workers, target_classes, transform_label=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = CIFAR10(root=dataset_dir, train=True, transform=transform, download=True)
    train_targets = np.array(train_dataset.targets)
    idx = np.isin(train_targets, target_classes)
    target_label = train_targets[idx].tolist()
    if transform_label:
        trans_label = [target_classes.index(i) for i in target_label]
        train_dataset.targets = trans_label
    else:
        train_dataset.targets = target_label
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = CIFAR10(root=dataset_dir, train=False,
                           transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_targets = np.array(test_dataset.targets)
    idx = np.isin(test_targets, target_classes)
    target_label = test_targets[idx].tolist()
    if transform_label:
        trans_label = [target_classes.index(i) for i in target_label]
        test_dataset.targets = trans_label
    else:
        test_dataset.targets = target_label
    test_dataset.data = test_dataset.data[idx]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_svhn(dataset_dir, batch_size, num_workers):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = SVHN(root=dataset_dir, split='train', transform=transform, download=True)
    unique_labels = np.unique(train_dataset.labels).tolist()
    train_dataset.classes = unique_labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = SVHN(root=dataset_dir, split='test',
                        transform=transforms.Compose([transforms.ToTensor(), normalize]), download=True)
    test_dataset.classes = unique_labels
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def load_svhn_target_class(dataset_dir, batch_size, num_workers, target_classes, transform_label=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])
    train_dataset = SVHN(root=dataset_dir, split='train', transform=transform, download=True)
    unique_labels = np.unique(train_dataset.labels).tolist()
    train_dataset.classes = unique_labels

    train_labels = train_dataset.labels
    idx = np.isin(train_labels, target_classes)
    target_labels = train_labels[idx].tolist()
    if transform_label:
        trans_labels = np.array([target_classes.index(i) for i in target_labels])
        train_dataset.labels = trans_labels
    else:
        train_dataset.labels = target_labels
    train_dataset.data = train_dataset.data[idx]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    test_dataset = SVHN(root=dataset_dir, split='test',
                        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_labels = test_dataset.labels
    idx = np.isin(test_labels, target_classes)
    target_labels = test_labels[idx].tolist()
    if transform_label:
        trans_labels = np.array([target_classes.index(i) for i in target_labels])
        test_dataset.labels = trans_labels
    else:
        test_dataset.labels = target_labels
    test_dataset.data = test_dataset.data[idx]
    test_dataset.classes = unique_labels
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


class ImageNetFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        self.imgs = self.samples
    
    @property
    def data(self):
        return np.array([path for path, _ in self.samples])
    
    @data.setter
    def data(self, new_data):
        if not hasattr(self, '_path_to_label'):
            self._path_to_label = {path: label for path, label in self.samples}
        self.samples = [(path, self._path_to_label[path]) for path in new_data]
        self.imgs = self.samples
    
    @property
    def targets(self):
        return np.array([label for _, label in self.samples])
    
    @targets.setter
    def targets(self, new_targets):
        assert len(new_targets) == len(self.samples), "New targets must match the number of samples."
        self.samples = [(path, target) for (path, _), target in zip(self.samples, new_targets)]
        self.imgs = self.samples

def load_imagenet(dataset_dir, batch_size, num_workers):
    dataset_dir = f"{dataset_dir}/imagenet"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform   = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageNetFolder(root=str(Path(dataset_dir)/'train'),
                                   transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)

    val_dataset = ImageNetFolder(root=str(Path(dataset_dir)/'val'),
                                 transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader

def load_imagenet_target_class(dataset_dir, batch_size, num_workers,
                               target_classes, transform_label=True):
    dataset_dir = f"{dataset_dir}/imagenet"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform   = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageNetFolder(root=str(Path(dataset_dir)/'train'),
                                   transform=train_transform)
    t = np.array(train_dataset.targets)
    idx = np.isin(t, target_classes)
    labs = t[idx].tolist()
    train_dataset.data = train_dataset.data[idx]
    train_dataset.targets = ([target_classes.index(l) for l in labs]
                             if transform_label else labs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)

    val_dataset = ImageNetFolder(root=str(Path(dataset_dir)/'val'),
                                 transform=val_transform)
    t = np.array(val_dataset.targets)
    idx = np.isin(t, target_classes)
    labs = t[idx].tolist()
    val_dataset.data = val_dataset.data[idx]
    val_dataset.targets = ([target_classes.index(l) for l in labs]
                           if transform_label else labs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)

    return train_loader, val_loader


if __name__ == '__main__':
    for tc in range(10):
        dataset = load_svhn_target_class(dataset_dir='../data/dataset/svhn', batch_size=128,
                                         num_workers=0, target_classes=[tc])
        print(f'tc_{tc} = {len(dataset[0])}')
