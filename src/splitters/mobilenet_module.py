import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mobilenet import MobileNet, Block


class MobileNet_Module(MobileNet):
    """
    MobileNet module that inherits from MobileNet and modifies layer dimensions
    based on active channel configurations.
    """
    
    def __init__(self, conv_channels, num_classes=10):
        """
        Initialize MobileNet module with reduced channel dimensions.
        
        Args:
            conv_channels: List of active channel counts for each conv layer
            num_classes: Number of output classes
        """
        # Initialize parent MobileNet first
        super().__init__(num_classes=num_classes)
        
        # Enable modular functionality
        self.is_modular = True
        
        # Store channel configuration
        self.conv_channels = conv_channels
        
        # Recreate layers with new dimensions
        self._recreate_layers_with_new_dims(conv_channels)
        
        # Add module head
        self.module_head = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, 1),
        )
    
    def _recreate_layers_with_new_dims(self, channels):
        """Recreate MobileNet layers with reduced channel dimensions by iterating through existing layers"""
        
        # Recreate conv1 and bn1 with new dimensions
        self.conv1 = self._recreate_conv_layer(self.conv1, out_channels=channels[0])
        self.bn1 = self._recreate_bn_layer(self.bn1, num_features=channels[0])
        
        # Recreate all blocks in self.layers with new dimensions
        new_layers = []
        for i, block in enumerate(self.layers):
            if i < len(channels) - 1:
                in_channels = channels[i]
                out_channels = channels[i + 1]
                new_block = self._recreate_mobile_block(block, in_channels, out_channels)
                new_layers.append(new_block)
        
        self.layers = nn.Sequential(*new_layers)
        
        # Recreate final linear layer
        self.linear = self._recreate_linear_layer(self.linear, in_features=channels[-1])
    
    def _recreate_conv_layer(self, original_conv, in_channels=None, out_channels=None):
        """Recreate a Conv2d layer with new dimensions while preserving other properties"""
        return nn.Conv2d(
            in_channels=in_channels or original_conv.in_channels,
            out_channels=out_channels or original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=in_channels if original_conv.groups == original_conv.in_channels else original_conv.groups,  # Handle depthwise conv
            bias=original_conv.bias is not None
        )
    
    def _recreate_bn_layer(self, original_bn, num_features):
        """Recreate a BatchNorm2d layer with new dimensions"""
        return nn.BatchNorm2d(
            num_features=num_features,
            eps=original_bn.eps,
            momentum=original_bn.momentum,
            affine=original_bn.affine,
            track_running_stats=original_bn.track_running_stats
        )
    
    def _recreate_linear_layer(self, original_linear, in_features=None, out_features=None):
        """Recreate a Linear layer with new dimensions"""
        return nn.Linear(
            in_features=in_features or original_linear.in_features,
            out_features=out_features or original_linear.out_features,
            bias=original_linear.bias is not None
        )
    
    def _recreate_mobile_block(self, original_block, in_channels, out_channels):
        """Recreate a MobileNet block with new channel dimensions using existing Block class"""
        
        # Determine stride from original block
        stride = 1
        if hasattr(original_block, 'conv1') and hasattr(original_block.conv1, 'stride'):
            stride = original_block.conv1.stride[0] if isinstance(original_block.conv1.stride, tuple) else original_block.conv1.stride
        
        # Create new Block with custom dimensions
        new_block = Block(in_planes=in_channels, out_planes=out_channels, stride=stride)
        
        # Override the layers with our custom dimensions (the Block class creates them correctly already)
        # We just need to ensure they match our expected dimensions
        new_block.conv1 = self._recreate_conv_layer(
            original_block.conv1, 
            in_channels=in_channels, 
            out_channels=in_channels  # Depthwise conv keeps same in/out channels
        )
        new_block.bn1 = self._recreate_bn_layer(original_block.bn1, in_channels)
        
        new_block.conv2 = self._recreate_conv_layer(
            original_block.conv2,
            in_channels=in_channels,
            out_channels=out_channels
        )
        new_block.bn2 = self._recreate_bn_layer(original_block.bn2, out_channels)
        
        return new_block
