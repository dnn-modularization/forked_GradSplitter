import numpy as np
import torch
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_module(module_conv_info, module_head_para, trained_model):
    if trained_model.__class__.__name__ == 'SimCNN':
        from models.simcnn import SimCNN
        module, module_param = _extract_module_simcnn(module_conv_info, module_head_para, trained_model, SimCNN)
    elif trained_model.__class__.__name__ == 'ResNet':
        from models.resnet18 import ResNet18
        module, module_param = _extract_module_resnet18(module_conv_info, module_head_para, trained_model)
    elif trained_model.__class__.__name__ == 'MobileNet':
        from models.mobilenet import MobileNet
        module, module_param = _extract_module_mobilenet(module_conv_info, module_head_para, trained_model)
    else:
        raise ValueError
    return module, module_param


def _extract_module_simcnn(module_conv_info, module_head_para, trained_model, model_network):
    """Extract module for SimCNN architectures."""
    conv_configs = []
    cin = 3
    for each_conv_layer in module_conv_info:
        n_kernels = each_conv_layer.size
        conv_configs.append((cin, n_kernels))
        cin = n_kernels

    module = model_network(num_classes=trained_model.num_classes, conv_configs=conv_configs)

    active_kernel_param = {}
    model_param = trained_model.state_dict()
    for i in range(len(conv_configs)):
        conv_weight = model_param[f'conv_{i}.0.weight']
        conv_bias = model_param[f'conv_{i}.0.bias']
        bn_weight = model_param[f'conv_{i}.1.weight']
        bn_bias = model_param[f'conv_{i}.1.bias']
        bn_running_mean = model_param[f'conv_{i}.1.running_mean']
        bn_running_var = model_param[f'conv_{i}.1.running_var']

        cur_conv_active_kernel_idx = module_conv_info[i]
        pre_conv_active_kernel_idx = module_conv_info[i-1] if i > 0 else list(range(3))

        tmp = conv_weight[cur_conv_active_kernel_idx, :, :, :]
        active_kernel_param[f'conv_{i}.0.weight'] = tmp[:, pre_conv_active_kernel_idx, :, :]
        active_kernel_param[f'conv_{i}.0.bias'] = conv_bias[cur_conv_active_kernel_idx]

        active_kernel_param[f'conv_{i}.1.weight'] = bn_weight[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.bias'] = bn_bias[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_mean'] = bn_running_mean[cur_conv_active_kernel_idx]
        active_kernel_param[f'conv_{i}.1.running_var'] = bn_running_var[cur_conv_active_kernel_idx]

    assert model_param[f'fc_{len(conv_configs)}.weight'].size(1) == model_param[f'conv_{len(conv_configs)-1}.0.bias'].size(0)
    first_fc_weight = model_param[f'fc_{len(conv_configs)}.weight']
    pre_conv_active_kernel_idx = module_conv_info[-1]
    active_first_fc_weight = first_fc_weight[:, pre_conv_active_kernel_idx]
    active_kernel_param[f'fc_{len(conv_configs)}.weight'] = active_first_fc_weight

    model_param.update(active_kernel_param)
    model_param.update(module_head_para)
    module.load_state_dict(model_param)
    model_device = next(trained_model.parameters()).device
    module = module.to(model_device).eval()
    return module, model_param


def _extract_conv_bn_params(model_param, active_indices, layer_prefix):
    """Extract conv and bn parameters for given active indices."""
    params = {}
    
    if f'{layer_prefix}.weight' in model_param:
        params[f'{layer_prefix}.weight'] = model_param[f'{layer_prefix}.weight'][active_indices]
    if f'{layer_prefix}.bias' in model_param:
        params[f'{layer_prefix}.bias'] = model_param[f'{layer_prefix}.bias'][active_indices]
    
    return params


def _extract_bn_params(model_param, active_indices, layer_prefix):
    """Extract batch normalization parameters."""
    params = {}
    
    bn_keys = ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
    for key in bn_keys:
        param_name = f'{layer_prefix}.{key}'
        if param_name in model_param:
            if key == 'num_batches_tracked':
                params[param_name] = model_param[param_name]
            else:
                params[param_name] = model_param[param_name][active_indices]
    
    return params


def _extract_conv_block_params(model_param, curr_active, prev_active, layer_prefix):
    """Extract convolutional layer parameters with proper indexing for input/output channels."""
    params = {}
    
    conv_weight = model_param[f'{layer_prefix}.weight']
    if len(conv_weight.shape) == 4:
        params[f'{layer_prefix}.weight'] = conv_weight[curr_active, :, :, :][:, prev_active, :, :]
    elif len(conv_weight.shape) == 2:
        params[f'{layer_prefix}.weight'] = conv_weight[:, prev_active]
    
    if f'{layer_prefix}.bias' in model_param:
        params[f'{layer_prefix}.bias'] = model_param[f'{layer_prefix}.bias'][curr_active]
    
    return params

def _extract_linear_params(model_param, input_active_indices, layer_prefix):
    """Extract linear layer parameters."""
    params = {}
    
    if f'{layer_prefix}.weight' in model_param:
        linear_weight = model_param[f'{layer_prefix}.weight']
        params[f'{layer_prefix}.weight'] = linear_weight[:, input_active_indices]
    
    if f'{layer_prefix}.bias' in model_param:
        params[f'{layer_prefix}.bias'] = model_param[f'{layer_prefix}.bias']
    
    return params


def _extract_module_resnet18(module_conv_info, module_head_para, trained_model):
    """Extract ResNet18 module with active kernels considering mask sharing."""
    from splitters.resnet18_module import ResNet18_Module
    
    model_device = next(trained_model.parameters()).device
    conv_channels = [len(conv_info) for conv_info in module_conv_info]
    
    module = ResNet18_Module(conv_channels, num_classes=trained_model.num_classes)
    module = module.to(model_device)
    
    model_param = trained_model.state_dict()
    active_kernel_param = {}
    
    layer_mapping, input_mapping = _get_resnet18_structure()
    
    for i, layer_name in enumerate(layer_mapping):
        conv_indices = module_conv_info[i]
        
        if i == 0:
            input_indices = list(range(3))
        else:
            prev_layer_idx = input_mapping.get(i, i - 1)
            input_indices = module_conv_info[prev_layer_idx]
        
        conv_params = _extract_conv_block_params(model_param, conv_indices, input_indices, layer_name)
        active_kernel_param.update(conv_params)
        
        if 'downsample' in layer_name:
            bn_layer = layer_name[:-2] + '.1' if layer_name.endswith('.0') else layer_name + '.1'
        else:
            bn_layer = layer_name.replace('conv', 'bn')
        
        if f'{bn_layer}.weight' in model_param:
            bn_params = _extract_bn_params(model_param, conv_indices, bn_layer)
            active_kernel_param.update(bn_params)
    
    fc_input_indices = module_conv_info[-1]
    active_kernel_param.update(_extract_linear_params(model_param, fc_input_indices, 'fc'))
    
    active_kernel_param.update(module_head_para)
    
    module.load_state_dict(active_kernel_param, strict=True)
    module = module.eval()
    
    return module, active_kernel_param


def _get_resnet18_structure():
    """Get ResNet18 layer structure with input dependencies for mask sharing."""
    layer_names = [
        'conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2',
        'layer2.0.conv1', 'layer2.0.downsample.0', 'layer2.0.conv2', 'layer2.1.conv1', 'layer2.1.conv2',
        'layer3.0.conv1', 'layer3.0.downsample.0', 'layer3.0.conv2', 'layer3.1.conv1', 'layer3.1.conv2',
        'layer4.0.conv1', 'layer4.0.downsample.0', 'layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2'
    ]
    
    input_mapping = {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5, 8: 7, 9: 8, 10: 9,
        11: 9, 12: 10, 13: 12, 14: 13, 15: 14, 16: 14, 17: 15, 18: 17, 19: 18
    }
    
    return layer_names, input_mapping



def _extract_module_mobilenet(module_conv_info, module_head_para, trained_model):
    """Extract MobileNet module using dynamic layer discovery."""
    from splitters.mobilenet_module import MobileNet_Module
    
    model_device = next(trained_model.parameters()).device
    conv_channels = [len(conv_info) for conv_info in module_conv_info]
    model_param = trained_model.state_dict()
    active_kernel_param = {}
    
    conv1_layers = [name for name in model_param.keys() if name.startswith('conv1') and '.weight' in name]
    block_layers = [name for name in model_param.keys() if name.startswith('layers.') and '.conv' in name and '.weight' in name]
    linear_layers = [name for name in model_param.keys() if name.startswith('linear') and '.weight' in name]
    
    blocks = {}
    for layer_name in block_layers:
        parts = layer_name.split('.')
        if len(parts) >= 3:
            block_idx = int(parts[1])
            conv_type = parts[2]
            if block_idx not in blocks:
                blocks[block_idx] = {}
            blocks[block_idx][conv_type] = layer_name.replace('.weight', '')
    
    sorted_blocks = sorted(blocks.items())
    num_blocks = len(sorted_blocks)
    
    expected_masks = 1 + num_blocks
    if len(module_conv_info) != expected_masks:
        print(f"Warning: Expected {expected_masks} masks but got {len(module_conv_info)}. Proceeding with available masks.")
    
    module = MobileNet_Module(conv_channels, num_classes=trained_model.num_classes)
    
    mask_idx = 0
    conv1_active = module_conv_info[mask_idx]
    active_kernel_param.update(_extract_conv_block_params(model_param, conv1_active, list(range(3)), 'conv1'))
    active_kernel_param.update(_extract_bn_params(model_param, conv1_active, 'bn1'))
    mask_idx += 1
    
    for block_idx, block_convs in sorted_blocks:
        if mask_idx >= len(module_conv_info):
            break
            
        if block_idx == 0:
            dw_input_active = conv1_active
        else:
            dw_input_active = module_conv_info[mask_idx - 1]
        
        pw_output_active = module_conv_info[mask_idx]
        
        if 'conv1' in block_convs and f'{block_convs["conv1"]}.weight' in model_param:
            dw_layer_name = block_convs['conv1']
            dw_weight = model_param[f'{dw_layer_name}.weight']
            active_kernel_param[f'{dw_layer_name}.weight'] = dw_weight[dw_input_active, :, :, :]
            if f'{dw_layer_name}.bias' in model_param:
                active_kernel_param[f'{dw_layer_name}.bias'] = model_param[f'{dw_layer_name}.bias'][dw_input_active]
            
            dw_bn_name = dw_layer_name.replace('conv1', 'bn1')
            active_kernel_param.update(_extract_bn_params(model_param, dw_input_active, dw_bn_name))
        
        if 'conv2' in block_convs and f'{block_convs["conv2"]}.weight' in model_param:
            pw_layer_name = block_convs['conv2']
            active_kernel_param.update(_extract_conv_block_params(model_param, pw_output_active, dw_input_active, pw_layer_name))
            
            pw_bn_name = pw_layer_name.replace('conv2', 'bn2')
            active_kernel_param.update(_extract_bn_params(model_param, pw_output_active, pw_bn_name))
        
        mask_idx += 1
    
    if linear_layers and mask_idx > 0:
        last_conv_active = module_conv_info[min(mask_idx - 1, len(module_conv_info) - 1)]
        for linear_layer in linear_layers:
            linear_name = linear_layer.replace('.weight', '')
            active_kernel_param.update(_extract_linear_params(model_param, last_conv_active, linear_name))
    
    active_kernel_param.update(module_head_para)
    
    module.load_state_dict(active_kernel_param, strict=False)
    module = module.to(model_device).eval()
    return module, active_kernel_param


def get_target_module_info(modules_info, trained_model, target_class, handle_warning=True):
    if trained_model.__class__.__name__ == 'SimCNN':
        module_conv_info, module_head_para = get_target_module_info_for_simcnn(modules_info, target_class,
                                                                               handle_warning)
    elif trained_model.__class__.__name__ == 'ResNet':
        module_conv_info, module_head_para = get_target_module_info_for_resnet18(modules_info, target_class,
                                                                                 handle_warning, trained_model)
    elif trained_model.__class__.__name__ == 'MobileNet':
        module_conv_info, module_head_para = get_target_module_info_for_mobilenet(modules_info, target_class,
                                                                                  handle_warning)
    else:
        raise ValueError
    return module_conv_info, module_head_para


def get_target_module_info_for_simcnn(modules_info, target_class, handle_warning):
    module_conv_info = []
    module_head_para = OrderedDict()
    for conv_idx in range(len(modules_info)):
        layer_name = f'module_{target_class}_conv_{conv_idx}'
        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
        else:
            break

    if f'module_{target_class}_head.weight' in modules_info:
        module_head_para[f'module_head.weight'] = modules_info[f'module_{target_class}_head.weight']
        module_head_para[f'module_head.bias'] = modules_info[f'module_{target_class}_head.bias']
    elif f'module_{target_class}_head.0.weight' in modules_info:
        module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
        module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
        module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
        module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    else:
        raise KeyError
    return module_conv_info, module_head_para



def get_target_module_info_for_resnet18(modules_info, target_class, handle_warning, trained_model=None):
    """Extract module information for ResNet18 using GradSplitter's mask sharing logic."""
    module_conv_info = []
    module_head_para = OrderedDict()
    
    from splitters.resnet18_splitter import GradSplitter
    
    if trained_model is None:
        class DummyModel:
            def __init__(self):
                self.num_classes = 10  # fallback default, should not normally be used
            def parameters(self):
                return [torch.tensor([1.0], device='cpu')]
        model = DummyModel()
    else:
        model = trained_model
    
    splitter = GradSplitter(model, module_init_type='zeros')
    splitter.load_state_dict(modules_info, strict=False)
    
    for layer_idx in range(20):
        try:
            mask = splitter.get_mask_for_layer(target_class, layer_idx)
            active_indices = torch.where(mask == 1)[0].cpu().numpy()
            if active_indices.size == 0 and handle_warning:
                active_indices = np.array([0]).astype('int64')
            module_conv_info.append(active_indices)
        except (AttributeError, KeyError) as e:
            if handle_warning:
                print(f"Warning: Could not get mask for layer {layer_idx}: {e}")
                module_conv_info.append(np.array([0]).astype('int64'))
            else:
                module_conv_info.append(np.array([]))
    
    head_prefix = f'module_{target_class}_head.'
    for param_name, param_value in modules_info.items():
        if param_name.startswith(head_prefix):
            relative_name = param_name[len(head_prefix):]
            module_head_para[f'module_head.{relative_name}'] = param_value
    
    return module_conv_info, module_head_para


def get_target_module_info_for_mobilenet(modules_info, target_class, handle_warning):
    """Extract module information for MobileNet architecture dynamically."""
    module_conv_info = []
    module_head_para = OrderedDict()
    
    conv_idx = 0
    while True:
        layer_name = f'module_{target_class}_conv_{conv_idx}'
        if layer_name in modules_info:
            each_conv_info = modules_info[layer_name]
            if hasattr(each_conv_info, 'numpy'):
                each_conv_info = each_conv_info.numpy()
            idx_info = np.argwhere(each_conv_info == 1)
            if idx_info.size == 0 and handle_warning:
                idx_info = np.array([[0]]).astype('int64')
            module_conv_info.append(np.squeeze(idx_info, axis=-1))
            conv_idx += 1
        else:
            break
    
    module_head_para.update(_extract_module_head_params(modules_info, target_class))
    
    return module_conv_info, module_head_para


def _extract_module_head_params(modules_info, target_class):
    """Extract module head parameters."""
    module_head_para = OrderedDict()
    
    if f'module_{target_class}_head.weight' in modules_info:
        module_head_para[f'module_head.weight'] = modules_info[f'module_{target_class}_head.weight']
        module_head_para[f'module_head.bias'] = modules_info[f'module_{target_class}_head.bias']
    elif f'module_{target_class}_head.0.weight' in modules_info:
        module_head_para[f'module_head.0.weight'] = modules_info[f'module_{target_class}_head.0.weight']
        module_head_para[f'module_head.0.bias'] = modules_info[f'module_{target_class}_head.0.bias']
        module_head_para[f'module_head.2.weight'] = modules_info[f'module_{target_class}_head.2.weight']
        module_head_para[f'module_head.2.bias'] = modules_info[f'module_{target_class}_head.2.bias']
    else:
        raise KeyError(f"Module head not found for class {target_class}")
    
    return module_head_para


def load_module(module_path, trained_model, target_class):
    modules_info = torch.load(module_path, map_location='cpu')
    module_conv_info, module_head_para = get_target_module_info(modules_info, trained_model, target_class)
    module, module_param = extract_module(module_conv_info, module_head_para, trained_model)
    return module


@torch.no_grad()
def module_predict(module, dataset):
    """Predict using the module on the given dataset."""
    module_device = next(module.parameters()).device
    
    outputs, labels = [], []
    for batch_inputs, batch_labels in dataset:
        batch_inputs = batch_inputs.to(module_device)
        batch_output = module(batch_inputs)
        outputs.append(batch_output)
        labels.append(batch_labels.to(module_device))
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    return outputs, labels

