import torch
import torch.nn as nn
from models.resnet18 import ResNet, BasicBlock


class ResNet18_Module(ResNet):
    """
    ResNet18 module that creates a ResNet with reduced channel dimensions
    based on active channel configurations from mask extraction.
    Inherits from ResNet and only overrides necessary parts.
    """
    
    def __init__(self, conv_channels, num_classes=10):
        """
        Initialize ResNet18 module with reduced channel dimensions.
        
        Args:
            conv_channels: List of active channel counts for each conv layer (20 values)
            num_classes: Number of output classes
        """
        if len(conv_channels) != 20:
            raise ValueError(f"Expected 20 channel values for ResNet18, got {len(conv_channels)}")
        
        # Call parent constructor with dummy parameters first
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        
        # Enable modular functionality
        self.is_modular = True
        self.conv_channels = conv_channels
        
        # Override layers with custom channel dimensions
        self._rebuild_with_custom_channels(conv_channels)
    
    def _rebuild_with_custom_channels(self, channels):
        """Rebuild only the layers that need custom channel dimensions."""
        # Use the exact channel dimensions from mask extraction
        # The corrected mask sharing in the splitter ensures ResNet constraints are met
        channels = list(channels)  # Make a copy to modify
        
        # Override initial conv
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        
        # Use exact channel dimensions from the corrected mask extraction
        # Layer 1: No downsampling
        self.layer1 = self._make_custom_layer(BasicBlock, 
                                             [(channels[0], channels[1], channels[2], 1, None),
                                              (channels[2], channels[3], channels[4], 1, None)])
        
        # Layer 2: First block has downsampling
        downsample2 = self._make_downsample(channels[4], channels[6], stride=2)
        self.layer2 = self._make_custom_layer(BasicBlock,
                                             [(channels[4], channels[5], channels[7], 2, downsample2),
                                              (channels[7], channels[8], channels[9], 1, None)])
        
        # Layer 3: First block has downsampling  
        downsample3 = self._make_downsample(channels[9], channels[11], stride=2)
        self.layer3 = self._make_custom_layer(BasicBlock,
                                             [(channels[9], channels[10], channels[12], 2, downsample3),
                                              (channels[12], channels[13], channels[14], 1, None)])
        
        # Layer 4: First block has downsampling
        downsample4 = self._make_downsample(channels[14], channels[16], stride=2)
        self.layer4 = self._make_custom_layer(BasicBlock,
                                             [(channels[14], channels[15], channels[17], 2, downsample4),
                                              (channels[17], channels[18], channels[19], 1, None)])
        
        # Override final linear layer - input from last conv layer
        self.fc = nn.Linear(channels[19], self.num_classes)
        
        # For modular functionality, don't set module_head = self.fc
        # The base ResNet class will handle the modular forward pass correctly
        # by applying fc first, then module_head if is_modular=True
        if self.is_modular:
            # Create the proper module_head that takes num_classes input
            self.module_head = nn.Sequential(
                nn.Linear(self.num_classes, self.num_classes),
                nn.ReLU(),
                nn.Linear(self.num_classes, 1),
            )
        
        # Re-initialize weights
        self._initialize_weights(zero_init_residual=False)
    
    def _make_custom_layer(self, block, block_configs):
        """Create a layer with custom block configurations."""
        layers = []
        for inplanes, conv1_planes, conv2_planes, stride, downsample in block_configs:
            layers.append(self._make_custom_block(block, inplanes, conv1_planes, conv2_planes, stride, downsample))
        return nn.Sequential(*layers)
    
    def _make_custom_block(self, block, inplanes, conv1_planes, conv2_planes, stride=1, downsample=None):
        """Create a BasicBlock with custom channel dimensions."""
        # Create new block instance
        custom_block = block(inplanes, conv1_planes, stride, downsample)
        
        # Override conv2 to have custom output channels
        custom_block.conv2 = nn.Conv2d(conv1_planes, conv2_planes, kernel_size=3, 
                                      stride=1, padding=1, bias=False)
        custom_block.bn2 = nn.BatchNorm2d(conv2_planes)
        
        return custom_block
    
    def _make_downsample(self, inplanes, planes, stride):
        """Create downsample layer (reuse parent functionality)."""
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )
    
    