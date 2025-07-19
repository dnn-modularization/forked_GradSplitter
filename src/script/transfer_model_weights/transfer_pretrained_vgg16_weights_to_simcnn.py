import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import re
from collections import defaultdict

from utils.configure_loader import load_configure
from utils.dataset_loader import get_dataset_loader
from utils.model_tools import DEVICE, evaluate, print_model_summary
from utils.model_loader import load_model


if __name__ == "__main__":
    model_name = 'simcnn'

    # dataset_name = 'cifar10'
    # dataset_name = 'cifar100'
    dataset_name = 'svhn'

    print(f"Transferring weights for {model_name} on {dataset_name}...")

    configs = load_configure(model_name, dataset_name)
    configs.set_estimator_idx()

    src_weights = torch.load(os.path.join(configs.data_dir, "source_model_weights", f"vgg16_{dataset_name}_std_model.pt"))
    model = load_model(model_name, configs.num_classes)
    print("-"*50)
    print_model_summary(model)

    base_weights = model.state_dict()
    tgt_weights = {k: None for k in base_weights.keys()}

    skipped = []
    for (s_k, s_v), (t_k, t_v) in zip(src_weights.items(), base_weights.items()):
        if s_v.shape == t_v.shape:
            tgt_weights[t_k] = s_v
        else:
            print(f"Skipping weight transfer from {s_k} to {t_k}, shape mismatch: {s_v.shape} vs {t_v.shape}")
            skipped.append((s_k, t_k))

    assert len(skipped) == 0, "Weights are skipped, something is wrong! Skipped: {}".format(skipped)

    missing = [k for k, v in tgt_weights.items() if v is None]
    assert len(missing) == 0, "Weights are not mapped, something is wrong! Missing: {}".format(missing)

    model.to(DEVICE)
    model.load_state_dict(tgt_weights)

    load_dataset = get_dataset_loader(dataset_name)
    train_loader, test_dataloader = load_dataset(configs.dataset_dir, batch_size=128, num_workers=2)

    print("-"*50)
    print_model_summary(model)
    
    print("-"*50)
    train_accuracy, train_per_class_accuracy = evaluate(model, train_loader, acc_in_percent=True)
    print(f"Train accuracy: {train_accuracy:.2f} - Per class accuracy: {[f'{acc:.2f}' for acc in train_per_class_accuracy]}")
    print("-"*50)

    test_accuracy, test_per_class_accuracy = evaluate(model, test_dataloader, acc_in_percent=True)
    print(f"Test accuracy: {test_accuracy:.2f} - Per class accuracy: {[f'{acc:.2f}' for acc in test_per_class_accuracy]}")
    print("-"*50)

    os.makedirs(os.path.dirname(configs.trained_model_path), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(configs.trained_model_path))
    print("Model weights transferred successfully! Saved to:", configs.trained_model_path)
    
