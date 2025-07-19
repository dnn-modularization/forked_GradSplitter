import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model
from utils.module_tools import load_module, module_predict
from utils.dataset_loader import get_dataset_loader_target_class
from utils.model_tools import count_parameters
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def cache_model_outputs(model, full_dataset):
    model.to(device)
    model.eval()
    
    all_outputs = []
    all_labels = []
    
    for inputs, labels in full_dataset:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_outputs.append(outputs.cpu())
        all_labels.append(labels)
    
    cached_model_outputs = torch.cat(all_outputs, dim=0)
    cached_model_labels = torch.cat(all_labels, dim=0)
    
    return cached_model_outputs, cached_model_labels

@torch.no_grad()
def evaluate_model_from_cache(cached_outputs, cached_labels, target_classes):
    task_mask = torch.isin(cached_labels, torch.tensor(target_classes))
    filtered_outputs = cached_outputs[task_mask]
    filtered_labels = cached_labels[task_mask]
    
    filtered_outputs = filtered_outputs[:, target_classes]
    
    label_mapping = {orig_class: new_idx for new_idx, orig_class in enumerate(target_classes)}
    mapped_labels = torch.tensor([label_mapping[label.item()] for label in filtered_labels])
    
    predicts = torch.argmax(filtered_outputs, dim=1)
    n_correct = torch.sum((predicts == mapped_labels).float())
    total_labels = len(mapped_labels)
    
    return n_correct / total_labels

@torch.no_grad()
def cache_module_outputs(modules, full_dataset):
    cached_outputs = {}
    data_labels = None
    
    for module_idx, module in enumerate(modules):
        module.to(device)
        module.eval()
        outputs, labels = module_predict(module, full_dataset)
        cached_outputs[module_idx] = outputs
        if data_labels is None:
            data_labels = labels
        else:
            assert (data_labels == labels).all()
    
    return cached_outputs, data_labels

@torch.no_grad()
def evaluate_reusing_from_cache(cached_outputs, target_classes, data_labels):
    selected_outputs = [cached_outputs[class_idx] for class_idx in target_classes]
    modules_outputs = torch.cat(selected_outputs, dim=1)
    final_pred = torch.argmax(modules_outputs, dim=1)
    acc = torch.div(torch.sum(final_pred == data_labels), len(data_labels))
    return acc

def setup_and_cache(configs, trained_model, module_path):
    all_modules = []
    all_module_param_counts = []
    for i in range(configs.num_classes):
        module = load_module(module_path, trained_model, i)
        all_modules.append(module)
        all_module_param_counts.append(count_parameters(module))

    load_dataset_target_classes = get_dataset_loader_target_class(args.dataset)
    _, full_test_dataloader = load_dataset_target_classes(
        configs.dataset_dir, batch_size=512, num_workers=4, 
        target_classes=list(range(configs.num_classes)), transform_label=False
    )
    
    cached_outputs, full_data_labels = cache_module_outputs(all_modules, full_test_dataloader)
    cached_model_outputs, cached_model_labels = cache_model_outputs(trained_model, full_test_dataloader)
    
    return all_module_param_counts, cached_outputs, full_data_labels, cached_model_outputs, cached_model_labels

def main():
    estimator_idx = args.estimator_idx

    with open(args.composition_task_file_path) as f:
        composition_tasks = [list(map(int, line.strip().split())) for line in f if line.strip()]

    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(estimator_idx)
    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    total_model_params = count_parameters(trained_model)
    module_path = configs.best_module_path

    all_module_param_counts, cached_outputs, full_data_labels, cached_model_outputs, cached_model_labels = setup_and_cache(
        configs, trained_model, module_path)

    for target_classes in composition_tasks:
        target_classes = sorted(target_classes)

        model_accuracy = evaluate_model_from_cache(cached_model_outputs, cached_model_labels, target_classes)

        task_mask = torch.isin(full_data_labels, torch.tensor(target_classes).to(device))
        filtered_labels = full_data_labels[task_mask]
        
        label_mapping = {orig_class: new_idx for new_idx, orig_class in enumerate(target_classes)}
        mapped_labels = torch.tensor([label_mapping[label.item()] for label in filtered_labels]).to(device)
        
        filtered_cached_outputs = {}
        for i, class_idx in enumerate(target_classes):
            filtered_cached_outputs[i] = cached_outputs[class_idx][task_mask]

        acc = evaluate_reusing_from_cache(filtered_cached_outputs, list(range(len(target_classes))), mapped_labels)
        
        module_param_counts = [all_module_param_counts[i] for i in target_classes]
        sum_module_params = sum(module_param_counts) if module_param_counts else 0

        print(
            f"STD_MODEL_ACC: {model_accuracy*100:.2f} "
            f"- COM_MODEL_ACC: {acc*100:.2f} "
            f"(Params: {sum_module_params:,}/{total_model_params:,}"
            f" ~ {sum_module_params / total_model_params:.2f}) "
            f"-------- {target_classes} "
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'resnet18', 'mobilenet'], default='simcnn')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'svhn'], default='cifar10')
    parser.add_argument('--estimator_idx', type=int, default=42)
    parser.add_argument('--composition_task_file_path', type=str, required=True,
                        help='Path to file containing composition tasks, one per line')
    args = parser.parse_args()
    print(args)
    main()
