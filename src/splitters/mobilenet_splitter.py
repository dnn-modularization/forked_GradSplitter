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

        # 14 independent masks control 27 conv layers (1 initial + 13 blocks × 2 layers each)
        # Depthwise layers share masks with their input layer for channel consistency
        self.conv_configs = [
            32,    # conv1
            64,    # block0 pointwise  
            128,   # block1 pointwise
            128,   # block2 pointwise
            256,   # block3 pointwise
            256,   # block4 pointwise
            512,   # block5 pointwise
            512,   # block6 pointwise
            512,   # block7 pointwise
            512,   # block8 pointwise
            512,   # block9 pointwise
            512,   # block10 pointwise
            1024,  # block11 pointwise
            1024,  # block12 pointwise
        ]
        
        # Depthwise mask sharing: depthwise_shared_masks[block_idx] = shared_mask_idx
        self.depthwise_shared_masks = {i: i for i in range(len(self.conv_configs) - 1)}

        self.module_params = []
        self.init_modules(module_init_type)

    def init_modules(self, module_init_type):
        for module_idx in range(self.n_modules):
            for layer_idx in range(len(self.conv_configs)):
                if module_init_type == 'random':
                    param = torch.randn(self.conv_configs[layer_idx]).to(device)
                elif module_init_type == 'ones':
                    param = torch.ones(self.conv_configs[layer_idx]).to(device)
                elif module_init_type == 'zeros':
                    param = torch.zeros(self.conv_configs[layer_idx]).to(device)
                else:
                    raise ValueError

                setattr(self, f'module_{module_idx}_conv_{layer_idx}', nn.Parameter(param))

            # Multi-layer head
            param = nn.Sequential(
                nn.Linear(self.n_class, self.n_class),
                nn.ReLU(),
                nn.Linear(self.n_class, 1),
            ).to(device)

            setattr(self, f'module_{module_idx}_head', param)
        print(getattr(self, f'module_{0}_head'))

    def forward(self, inputs):
        predicts = []
        for module_idx in range(self.n_modules):
            each_module_pred = self.module_predict(inputs, module_idx)
            predicts.append(each_module_pred)
        predicts = torch.cat(predicts, dim=1)
        return predicts

    def module_predict(self, x, module_idx):
        """
        Make prediction using the specified module.
        Uses get_intermediate_features and applies the multi-layer head.
        """
        # Get intermediate features (before multi-layer head)
        pred = self.get_intermediate_features(x, module_idx)
        
        # Module head
        module_head = getattr(self, f'module_{module_idx}_head')
        pred = torch.relu(pred)
        head_output = torch.sigmoid(module_head(pred))
        return head_output

    def get_intermediate_features(self, x, module_idx):
        """
        Get intermediate features before the multi-layer head.
        This is the core implementation used by both module_predict and testing.
        """
        # Initial conv + bn + relu
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu1(x)
        
        # Apply module mask for conv1 (mask_idx = 0)
        layer_param = getattr(self, f'module_{module_idx}_conv_0')
        layer_param_proc = self.sign(layer_param)
        x = torch.einsum('j, ijkl->ijkl', layer_param_proc, x)
        
        # Process each block in the layers
        for block_idx, block in enumerate(self.model.layers):
            # Depthwise conv - shares mask with previous layer
            out = block.conv1(x)
            out = block.bn1(out)
            out = block.relu1(out)
            
            # Apply shared mask for depthwise conv (shares with input layer)
            shared_mask_idx = self.depthwise_shared_masks[block_idx]
            layer_param = getattr(self, f'module_{module_idx}_conv_{shared_mask_idx}')
            layer_param_proc = self.sign(layer_param)
            out = torch.einsum('j, ijkl->ijkl', layer_param_proc, out)
            
            # Pointwise conv - has its own mask
            out = block.conv2(out)
            out = block.bn2(out)
            out = block.relu2(out)
            
            # Apply module mask for pointwise conv (independent mask)
            pointwise_mask_idx = block_idx + 1  # mask indices 1-13 for pointwise layers
            layer_param = getattr(self, f'module_{module_idx}_conv_{pointwise_mask_idx}')
            layer_param_proc = self.sign(layer_param)
            out = torch.einsum('j, ijkl->ijkl', layer_param_proc, out)
            
            x = out
        
        # Final layers - stop before multi-layer head
        x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        pred = self.model.linear(x)
        
        return pred  # Return raw logits, not processed through multi-layer head

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
        Goes layer by layer through MobileNet structure, applying appropriate masks.
        """
        module_used_kernels = []
        for module_idx in range(self.n_modules):
            each_module_kernels = []
            
            # Layer 0: conv1 - uses mask_0
            layer_param = getattr(self, f'module_{module_idx}_conv_0')
            each_module_kernels.append(self.sign(layer_param))
            
            # Remaining layers: blocks × 2 layers each (depthwise + pointwise)
            num_blocks = len(self.conv_configs) - 1  # exclude conv1
            for block_idx in range(num_blocks):
                # Depthwise layer - shares mask with previous layer
                shared_mask_idx = self.depthwise_shared_masks[block_idx]
                layer_param = getattr(self, f'module_{module_idx}_conv_{shared_mask_idx}')
                each_module_kernels.append(self.sign(layer_param))
                
                # Pointwise layer - has its own independent mask
                pointwise_mask_idx = block_idx + 1
                layer_param = getattr(self, f'module_{module_idx}_conv_{pointwise_mask_idx}')
                each_module_kernels.append(self.sign(layer_param))
            
            module_used_kernels.append(torch.cat(each_module_kernels))
        return torch.stack(module_used_kernels)


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
