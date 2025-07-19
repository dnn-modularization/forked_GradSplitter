import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GradSplitter(nn.Module):
    def __init__(self, model, module_init_type):
        super(GradSplitter, self).__init__()
        self.model = model
        self.n_class = model.num_classes
        self.n_modules = self.n_class
        self.sign = MySign.apply
        for p in model.parameters():
            p.requires_grad = False

        # ResNet18 layer configuration:
        # conv1: 64 channels
        # layer1: 2 blocks, each with conv1(64->64) + conv2(64->64), no downsample
        # layer2: 2 blocks, each with conv1(64->128) + conv2(128->128), first block has downsample(64->128)
        # layer3: 2 blocks, each with conv1(128->256) + conv2(256->256), first block has downsample(128->256)  
        # layer4: 2 blocks, each with conv1(256->512) + conv2(512->512), first block has downsample(256->512)
        self.conv_configs = [
            64,    # 0: conv1
            64,    # 1: layer1.0.conv1 
            64,    # 2: layer1.0.conv2 (must match input for residual)
            64,    # 3: layer1.1.conv1
            64,    # 4: layer1.1.conv2 (must match input for residual)
            128,   # 5: layer2.0.conv1
            128,   # 6: layer2.0.downsample
            128,   # 7: layer2.0.conv2 (must match downsample for residual)
            128,   # 8: layer2.1.conv1
            128,   # 9: layer2.1.conv2 (must match input for residual)
            256,   # 10: layer3.0.conv1
            256,   # 11: layer3.0.downsample
            256,   # 12: layer3.0.conv2 (must match downsample for residual)
            256,   # 13: layer3.1.conv1
            256,   # 14: layer3.1.conv2 (must match input for residual)
            512,   # 15: layer4.0.conv1
            512,   # 16: layer4.0.downsample
            512,   # 17: layer4.0.conv2 (must match downsample for residual)
            512,   # 18: layer4.1.conv1
            512,   # 19: layer4.1.conv2 (must match input for residual)
        ]
        
        # Independent mask indices (layers that have their own unique masks)
        self.independent_mask_indices = [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18]
        
        # Mask sharing for ResNet constraints
        # Format: {shared_layer_idx: provider_layer_idx}
        self.shared_mask_mapping = {
            # No downsample blocks: conv2 must match block input
            2: 0,   # layer1.0.conv2 shares with conv1 (block input)
            4: 2,   # layer1.1.conv2 shares with layer1.0.conv2 (block input)
            9: 7,   # layer2.1.conv2 shares with layer2.0.conv2 (block input)
            14: 12, # layer3.1.conv2 shares with layer3.0.conv2 (block input)
            19: 17, # layer4.1.conv2 shares with layer4.0.conv2 (block input)
            
            # Downsample blocks: conv2 must match downsample output
            7: 6,   # layer2.0.conv2 shares with layer2.0.downsample
            12: 11, # layer3.0.conv2 shares with layer3.0.downsample
            17: 16, # layer4.0.conv2 shares with layer4.0.downsample
        }

        self.module_params = []
        self.init_modules(module_init_type)

    def init_modules(self, module_init_type):
        for module_idx in range(self.n_modules):
            # Initialize only independent masks
            for mask_idx in self.independent_mask_indices:
                if module_init_type == 'random':
                    param = torch.randn(self.conv_configs[mask_idx])
                elif module_init_type == 'ones':
                    param = torch.ones(self.conv_configs[mask_idx])
                elif module_init_type == 'zeros':
                    param = torch.zeros(self.conv_configs[mask_idx])
                else:
                    raise ValueError(f"Unknown module_init_type: {module_init_type}")

                # Move to same device as model
                param = param.to(next(self.model.parameters()).device)
                setattr(self, f'module_{module_idx}_conv_{mask_idx}', nn.Parameter(param))

            # Multi-layer head
            param = nn.Sequential(
                nn.Linear(self.n_class, self.n_class),
                nn.ReLU(),
                nn.Linear(self.n_class, 1),
            )
            # Move to same device as model
            param = param.to(next(self.model.parameters()).device)
            setattr(self, f'module_{module_idx}_head', param)

    def get_mask_for_layer(self, module_idx, layer_idx):
        """Get the appropriate mask for a given layer, considering mask sharing."""
        # Resolve the mask sharing chain to find the ultimate provider
        ultimate_provider = layer_idx
        visited = set()
        
        while ultimate_provider in self.shared_mask_mapping:
            if ultimate_provider in visited:
                raise ValueError(f"Circular mask sharing detected for layer {layer_idx}")
            visited.add(ultimate_provider)
            ultimate_provider = self.shared_mask_mapping[ultimate_provider]
        
        # Get the mask from the ultimate provider
        layer_param = getattr(self, f'module_{module_idx}_conv_{ultimate_provider}')
        return self.sign(layer_param)

    def forward(self, inputs):
        predicts = []
        for module_idx in range(self.n_modules):
            each_module_pred = self.module_predict(inputs, module_idx)
            predicts.append(each_module_pred)
        predicts = torch.cat(predicts, dim=1)
        return predicts

    def module_predict(self, x, module_idx):
        """Make prediction using the specified module."""
        # Get intermediate features (before multi-layer head)
        pred = self.get_intermediate_features(x, module_idx)
        
        # Module head
        module_head = getattr(self, f'module_{module_idx}_head')
        pred = torch.relu(pred)
        head_output = torch.sigmoid(module_head(pred))
        return head_output

    def get_intermediate_features(self, x, module_idx):
        """Get intermediate features before the multi-layer head."""
        layer_idx = 0
        
        # Initial conv + bn + relu
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu1(x)
        
        # Apply mask for conv1
        layer_param_proc = self.get_mask_for_layer(module_idx, layer_idx)
        x = torch.einsum('j, ijkl->ijkl', layer_param_proc, x)
        layer_idx += 1
        
        # Process each layer group
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer_group = getattr(self.model, layer_name)
            
            for block_idx, block in enumerate(layer_group):
                identity = x
                
                # Conv1 of the block
                out = block.conv1(x)
                out = block.bn1(out)
                out = block.relu1(out)
                
                # Apply mask for conv1
                layer_param_proc = self.get_mask_for_layer(module_idx, layer_idx)
                out = torch.einsum('j, ijkl->ijkl', layer_param_proc, out)
                layer_idx += 1
                
                # Handle downsample if present (only for first block of layer2, layer3, layer4)
                if block.downsample is not None:
                    # Apply downsample to identity
                    identity = block.downsample(identity)
                    
                    # Apply mask for downsample
                    layer_param_proc = self.get_mask_for_layer(module_idx, layer_idx)
                    identity = torch.einsum('j, ijkl->ijkl', layer_param_proc, identity)
                    layer_idx += 1
                
                # Conv2 of the block
                out = block.conv2(out)
                out = block.bn2(out)
                
                # Apply mask for conv2 (shares mask with downsample or conv1)
                layer_param_proc = self.get_mask_for_layer(module_idx, layer_idx)
                out = torch.einsum('j, ijkl->ijkl', layer_param_proc, out)
                layer_idx += 1
                
                # Residual connection
                out += identity
                out = block.relu2(out)
                
                x = out
        
        # Final layers
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        pred = self.model.fc(x)
        
        return pred

    def get_module_params(self):
        module_params = OrderedDict()
        total_params = self.state_dict()
        for layer_name in total_params:
            if layer_name.startswith('module'):
                if 'conv' in layer_name:
                    module_params[layer_name] = (total_params[layer_name] > 0).int()
                else:
                    module_params[layer_name] = total_params[layer_name]
        return module_params

    def get_module_kernels(self):
        """
        Returns kernel usage for each module.
        Note: This counts unique active kernels, not total 1's in masks (due to mask sharing).
        """
        module_used_kernels = []
        for module_idx in range(self.n_modules):
            each_module_kernels = []
            
            # Go through all 20 conv layers and get their effective masks
            for layer_idx in range(len(self.conv_configs)):
                layer_param_proc = self.get_mask_for_layer(module_idx, layer_idx)
                each_module_kernels.append(layer_param_proc)
            
            module_used_kernels.append(torch.cat(each_module_kernels))
        
        return torch.stack(module_used_kernels)


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
