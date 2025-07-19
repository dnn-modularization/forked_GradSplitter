import argparse
import sys
import torch
from pathlib import Path
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model
from utils.module_tools import load_module

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_model_flops(model, sample_input):
    """Compute FLOPs for a model using FlopCountAnalysis."""
    model.eval()
    flop_count = FlopCountAnalysis(model, sample_input).unsupported_ops_warnings(False)
    return flop_count.total()

def main():
    estimator_idx = args.estimator_idx

    # Read composition tasks from file
    with open(args.composition_task_file_path) as f:
        composition_tasks = [list(map(int, line.strip().split())) for line in f if line.strip()]

    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(estimator_idx)
    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    trained_model.to(DEVICE)
    module_path = configs.best_module_path

    # Create sample input for FLOP analysis
    sample_input_batch = torch.randn(1, 3, 32, 32, device=DEVICE)

    # Compute FLOPs for the standard model
    std_model_flops = compute_model_flops(trained_model, sample_input_batch)

    # Pre-compute FLOPs for all individual modules
    print("Pre-computing FLOPs for all modules...")
    all_module_flops = []
    for i in range(configs.num_classes):
        module = load_module(module_path, trained_model, i)
        module.to(DEVICE)
        module_flops = compute_model_flops(module, sample_input_batch)
        all_module_flops.append(module_flops)
        # Clean up to save memory
        del module
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Standard model FLOPs: {std_model_flops:,}")
    print("Evaluating composition tasks...")

    for target_classes in composition_tasks:
        target_classes = sorted(target_classes)
        
        # Sum FLOPs for selected modules
        selected_module_flops = [all_module_flops[i] for i in target_classes]
        total_composed_flops = sum(selected_module_flops)
        
        # Calculate percentage
        flop_percentage = (total_composed_flops / std_model_flops) if std_model_flops > 0 else 0

        print(
            f"COM_MODEL_FLOPS/STD_MODEL_FLOPS: "
            f"{total_composed_flops}/{std_model_flops}="
            f"{flop_percentage:.4f}  -------- {target_classes}"
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