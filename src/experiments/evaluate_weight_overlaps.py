import argparse
import sys
import torch
import numpy as np
import itertools
import random
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.configure_loader import load_configure
from utils.model_loader import load_trained_model, load_model
from utils.module_tools import get_target_module_info, extract_module
from utils.model_tools import count_parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_trackable_params(model_state_dict):
    """
    Generate unique float64 values for each parameter to track weight usage.
    This replaces actual weights with unique identifiers.
    """
    trackable_params = {}
    unique_number = 0
    
    for param_name, param_tensor in model_state_dict.items():
        # Create unique sequential IDs for trainable parameters
        numel = param_tensor.numel()
        new_tensor = torch.arange(unique_number, unique_number + numel, dtype=torch.float64, device=DEVICE)
        unique_number += numel
        trackable_params[param_name] = new_tensor.view(param_tensor.shape)
    
    
    all_flattened = torch.cat([p.flatten() for p in trackable_params.values()])
    unique_flattened = torch.unique(all_flattened)
    assert len(unique_flattened) == unique_number, (
        f"Expected {unique_number} unique values, got {len(unique_flattened)}"
    )
    return trackable_params, unique_number

def get_module_unique_weights(model_name, modules_info, target_class, trained_model, trackable_params):
    """
    Extract a module using trackable parameters and return the set of unique weight IDs and the module.
    """
    # Get module info and extract the module
    module_conv_info, module_head_para = get_target_module_info(modules_info, trained_model, target_class)

    # Create temporary model with trackable parameters
    temp_model = load_model(model_name, trained_model.num_classes)

    temp_model.double()
    temp_model.load_state_dict(trackable_params)
    temp_model.to(next(trained_model.parameters()).device)

    # Extract module with trackable parameters
    module, module_param = extract_module(module_conv_info, module_head_para, temp_model)

    # Collect all unique weight IDs from the module
    param_tensors = []
    for param_name, param_tensor in module_param.items():
        if param_name.startswith('module_head.'):
            # Skip head parameters, as it's constructed exclusively for each module
            continue
        param_tensors.append(param_tensor.view(-1))
    
    # Concatenate all tensors at once and convert to int in batch
    if param_tensors:
        all_params = torch.cat(param_tensors)
        flatten_param_list = all_params.int().tolist()
    else:
        flatten_param_list = []

    # Debug uniqueness: ensure all IDs are unique
    unique_weights = set(flatten_param_list)
    assert len(flatten_param_list) == len(unique_weights), (
        f"Non-unique IDs found in module {target_class}: "
        f"{len(set(flatten_param_list))} unique, "
        f"{len(flatten_param_list)} total"
    )
    
    return unique_weights, module

def calculate_overlap_params(modules_weights, total_model_params, num_classes):
    
    # Calculate all combinations of modules
    indices_combinations = list(itertools.combinations(range(len(modules_weights)), 2))
    if num_classes == 100:
        # For 100 classes, sample 357 combinations
        # 357 = sampling from population of [4950 indices_combinations] with confidence level of 95%, margin of error 5%
        indices_combinations = random.sample(indices_combinations, k=357)

    print(f"Calculating {len(indices_combinations)} pairwise overlaps...")
    overlap_results = []
    
    for module1_index, module2_index in indices_combinations:
        module1_params = modules_weights[module1_index]
        module2_params = modules_weights[module2_index]
        
        # Calculate intersection
        intersection = module1_params & module2_params
        overlap_count = len(intersection)
        overlap_percentage = (overlap_count / total_model_params) * 100
        
        overlap_results.append({
            'module1': module1_index,
            'module2': module2_index,
            'overlap_count': overlap_count,
            'overlap_percentage': overlap_percentage
        })
        print({
            'module1': module1_index,
            'module2': module2_index,
            'overlap_count': overlap_count,
            'overlap_percentage': overlap_percentage
        })
    
    return overlap_results

def main():
    parser = argparse.ArgumentParser(description='Calculate weight overlaps between GradSplitter modules')
    parser.add_argument('--model', choices=['simcnn', 'resnet18', 'mobilenet'], default='simcnn')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'svhn'], default='cifar10')
    parser.add_argument('--estimator_idx', type=int, default=42)
    
    args = parser.parse_args()
    
    print("Weight Overlap Calculator - GradSplitter Modules")
    print("=" * 60)
    
    # Load configuration and model
    configs = load_configure(args.model, args.dataset)
    configs.set_estimator_idx(args.estimator_idx)
    
    trained_model = load_trained_model(configs.model_name, configs.num_classes, configs.trained_model_path)
    trained_model.double()
    trained_model.to(DEVICE)
    
    modules_info = torch.load(configs.best_module_path, map_location='cpu')
    
    print("=" * 60)
    # Generate trackable parameters with unique IDs
    print("Generating trackable parameters with unique IDs...")
    model_state = trained_model.state_dict()
    trackable_params, total_model_params = generate_trackable_params(model_state)
    
    print(f"Total trainable parameters: {total_model_params:,}")
    print(f"Model: {args.model}, Dataset: {args.dataset}, Classes: {configs.num_classes}")
    
    print("\nExtracting weight sets for all modules...")
    print("=" * 60)
    
    # Extract unique weights for all modules
    modules_weights = []
    module_counts = []  # raw parameter counts
    
    for i in tqdm(range(configs.num_classes), desc="Processing modules"):
        module_weights, module = get_module_unique_weights(args.model, modules_info, i, trained_model, trackable_params)
        modules_weights.append(module_weights)
        
        module_size = count_parameters(module)
        module_counts.append(module_size)

        del module
    
    # Calculate overlaps (all combinations for 10 classes, sampled for 100 classes)
    print(f"\nCalculating overlaps...")
    overlap_results = calculate_overlap_params(modules_weights, total_model_params, configs.num_classes)
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS:")
    
    # Gather overlap counts
    overlap_counts = [r['overlap_count'] for r in overlap_results if r['overlap_count'] > 0]
    
    if overlap_counts:
        # Summary of average module size and overlap (percent derived)
        avg_mod_count = np.mean(module_counts)
        avg_mod_pct = (avg_mod_count / total_model_params) * 100
        avg_overlap_count = np.mean(overlap_counts)
        avg_overlap_pct = (avg_overlap_count / total_model_params) * 100
        
        print(f"Total original model size: {total_model_params:,}")
        print(f"Average module size: {avg_mod_count:.2f} ({avg_mod_pct:.2f}%)")
        print(f"Average overlap: {avg_overlap_count:.2f} ({avg_overlap_pct:.2f}%)")
        
    else:
        print("No overlaps found between any module pairs")

if __name__ == '__main__':
    main()
