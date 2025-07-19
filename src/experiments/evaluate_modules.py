import argparse
import sys
import torch
from pathlib import Path
# sys.path.append('')
# sys.path.append('..')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model
from utils.module_tools import load_module, module_predict
from utils.dataset_loader import get_dataset_loader
from utils.model_tools import count_parameters
from utils.splitter_loader import load_splitter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_module_f1(module, dataset, target_class):
    module.to(device)
    module.eval()
    outputs, labels = module_predict(module, dataset)
    predicts = (outputs > 0.5).int().squeeze(-1)
    labels = (labels == target_class).int()

    precision = torch.sum(predicts * labels) / torch.sum(predicts) if torch.sum(predicts) > 0 else torch.tensor(0.0)
    recall = torch.sum(predicts * labels) / torch.sum(labels) if torch.sum(labels) > 0 else torch.tensor(0.0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
    return f1

@torch.no_grad()
def evaluate_model_f1_per_class(model, dataset_loader, num_classes):
    model.to(device)
    model.eval()
    all_outputs = []
    all_labels = []
    for inputs, labels in dataset_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_outputs.append(outputs.detach().cpu())
        all_labels.append(labels.cpu())
    outputs = torch.cat(all_outputs, dim=0)
    labels = torch.cat(all_labels, dim=0)

    f1_scores = []
    for target_class in range(num_classes):
        predicts = (outputs.argmax(dim=1) == target_class).int()
        true_labels = (labels == target_class).int()
        precision = torch.sum(predicts * true_labels) / torch.sum(predicts) if torch.sum(predicts) > 0 else torch.tensor(0.0)
        recall = torch.sum(predicts * true_labels) / torch.sum(true_labels) if torch.sum(true_labels) > 0 else torch.tensor(0.0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
        f1_scores.append(f1.item())
    return f1_scores


def main():
    estimator_idx = args.estimator_idx
    print(f'Estimator {estimator_idx}')
    print('-' * 80)

    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(estimator_idx)

    dataset_dir = configs.dataset_dir
    load_dataset = get_dataset_loader(args.dataset)
    _ , test_dataloader = load_dataset(dataset_dir, batch_size=128, num_workers=2)

    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    total_model_params = count_parameters(trained_model)

    model_per_class_f1 = evaluate_model_f1_per_class(trained_model, test_dataloader, configs.num_classes)
    for i, model_f1 in enumerate(model_per_class_f1):
        print(f'[Original model] Class {i} F1-score: {model_f1:.4f}')
    print(f'[Original model] --- Average F1-score per class: {sum(model_per_class_f1) / len(model_per_class_f1):.4f}')
    print(f'[Original model] --- Total model parameters: {total_model_params:,}')

    # evaluate each module
    module_path = configs.best_module_path
    module_f1_scores = []
    module_param_counts = []
    
    for i in range(configs.num_classes):
        module = load_module(module_path, trained_model, i)
        total_module_params = count_parameters(module)
        module_param_counts.append(total_module_params)
        module_f1_score = evaluate_module_f1(module, test_dataloader, i)
        module_f1_scores.append(module_f1_score.item())
        print(f'[Module] Class {i} F1-score: {module_f1_score:.4f} - Total parameters: {total_module_params:,}')
    avg_f1 = sum(module_f1_scores) / len(module_f1_scores) if module_f1_scores else 0.0
    avg_params = sum(module_param_counts) / len(module_param_counts) if module_param_counts else 0.0
    print(f'[Module] --- Average module F1-score: {avg_f1:.4f}')
    print(f'[Module] --- Average module parameters: {avg_params:,.0f} ({avg_params / total_model_params:.2%} of original model)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['simcnn', 'resnet18', 'mobilenet'], default='resnet18')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'svhn'], default='cifar10')
    parser.add_argument('--estimator_idx', type=int, default=42)
    args = parser.parse_args()
    print(args)
    main()
