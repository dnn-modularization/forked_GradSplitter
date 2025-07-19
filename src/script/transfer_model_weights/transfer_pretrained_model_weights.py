import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from models.resnet18 import ResNet18
import re
from collections import defaultdict

from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.model_tools import DEVICE, evaluate, print_model_summary
from utils.model_loader import load_model


if __name__ == "__main__":
    model_name = 'resnet18'
    # model_name = 'mobilenet'
    
    # dataset_name = 'cifar10'
    dataset_name = 'svhn'
    # dataset_name = 'cifar100'

    print(f"Transferring weights for {model_name} on {dataset_name}...")

    configs = load_configure(model_name, dataset_name)
    configs.set_estimator_idx()

    src_weights = torch.load(os.path.join(configs.data_dir, "source_model_weights", f"{model_name}_{dataset_name}_std_model.pt"))
    model = load_model(model_name, configs.num_classes)
    model.load_state_dict(src_weights)

    model.to(DEVICE)

    load_dataset = get_dataset_loader(dataset_name)
    train_loader, test_dataloader = load_dataset(configs.dataset_dir, batch_size=128, num_workers=2)

    print("-"*50)
    print_model_summary(model)
    
    print("-"*50)
    train_accuracy, train_per_class_accuracy = evaluate(model, train_loader, acc_in_percent=True)
    train_per_class_accuracy_str = [f"{acc:.2f}" for acc in train_per_class_accuracy]
    print(f"Train accuracy: {train_accuracy:.2f} - Per class accuracy: {train_per_class_accuracy_str}")

    test_accuracy, test_per_class_accuracy = evaluate(model, test_dataloader, acc_in_percent=True)
    print(f"Test accuracy: {test_accuracy:.2f} - Per class accuracy: {[f'{acc:.2f}' for acc in test_per_class_accuracy]}")

    print("-"*50)

    os.makedirs(os.path.dirname(configs.trained_model_path), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(configs.trained_model_path))
    print("Model weights transferred successfully! Saved to:", configs.trained_model_path)
    
