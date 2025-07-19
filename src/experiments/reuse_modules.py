import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model
from utils.module_tools import load_module, module_predict
from utils.dataset_loader import get_dataset_loader_target_class
from utils.model_tools import count_parameters, evaluate_model_for_target_classes
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_reusing(modules, dataset):
    for m in modules:
        m.to(device)
        m.eval()
    
    modules_outputs = []
    data_labels = None
    for each_module in modules:
        outputs, labels = module_predict(each_module, dataset)
        modules_outputs.append(outputs)
        if data_labels is None:
            data_labels = labels
        else:
            assert (data_labels == labels).all()

    modules_outputs = torch.cat(modules_outputs, dim=1)
    final_pred = torch.argmax(modules_outputs, dim=1)
    acc = torch.div(torch.sum(final_pred == data_labels), len(data_labels))
    return acc

def main():
    estimator_idx = args.estimator_idx

    # Read composition tasks from file
    with open(args.composition_task_file_path) as f:
        composition_tasks = [list(map(int, line.strip().split())) for line in f if line.strip()]

    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(estimator_idx)
    dataset_dir = configs.dataset_dir
    load_dataset_target_classes = get_dataset_loader_target_class(args.dataset)
    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    total_model_params = count_parameters(trained_model)
    module_path = configs.best_module_path

    # Pre-load all modules
    all_modules = []
    all_module_param_counts = []
    for i in range(configs.num_classes):
        module = load_module(module_path, trained_model, i)
        all_modules.append(module)
        all_module_param_counts.append(count_parameters(module))

    for target_classes in composition_tasks:
        start_time = time.time()
        target_classes = sorted(target_classes)
        _, test_dataloader = load_dataset_target_classes(
            dataset_dir, batch_size=512, num_workers=4, target_classes=target_classes, transform_label=True
        )

        # Evaluate original model
        model_accuracy = evaluate_model_for_target_classes(trained_model, test_dataloader, target_classes)

        # Select pre-loaded modules for current task
        selected_modules = [all_modules[i] for i in target_classes]
        module_param_counts = [all_module_param_counts[i] for i in target_classes]

        # Evaluate reusing selected modules
        acc = evaluate_reusing(selected_modules, test_dataloader)
        sum_module_params = sum(module_param_counts) if module_param_counts else 0

        elapsed = time.time() - start_time

        print(
            f"STD_MODEL_ACC: {model_accuracy*100:.2f} "
            f"- COM_MODEL_ACC: {acc*100:.2f} "
            f"(Params: {sum_module_params:,}/{total_model_params:,}"
            f" ~ {sum_module_params / total_model_params:.2f}) "
            f"-------- {target_classes} "
            f"[Time: {elapsed:.2f}s]"
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
