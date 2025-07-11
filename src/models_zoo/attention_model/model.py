

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the base model components from the original model file
from ..base_model.model import _DenseLayer, _DenseBlock, _Transition

# Import the attention module
from .attention import CBAM

class MicroSegNetAttention(nn.Module):
    """
    MicroSegNet with CBAM Attention Gates in the decoder's skip connections.
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):
        super(MicroSegNetAttention, self).__init__()

        # --- Encoder (re-used from the original MicroSegNet) ---
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        self.encoder_blocks = nn.ModuleList()
        self.transition_blocks = nn.ModuleList()
        encoder_channels = [num_init_features]

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.encoder_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            encoder_channels.append(num_features)
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.transition_blocks.append(trans)
                num_features = num_features // 2
                encoder_channels.append(num_features)

        # --- Decoder with Attention Gates ---
        self.bottleneck_conv = nn.Conv2d(num_features, 512, kernel_size=1, stride=1, bias=False)
        
        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att1 = CBAM(encoder_channels[5])
        self.up_conv1 = self._make_up_conv_block(256 + encoder_channels[5], 256)

        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = CBAM(encoder_channels[3])
        self.up_conv2 = self._make_up_conv_block(128 + encoder_channels[3], 128)

        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att3 = CBAM(encoder_channels[1])
        self.up_conv3 = self._make_up_conv_block(64 + encoder_channels[1], 64)
        
        self.upsample4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.att4 = CBAM(encoder_channels[0])
        self.up_conv4 = self._make_up_conv_block(64 + encoder_channels[0], 64)
        
        self.upsample5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_conv5 = self._make_up_conv_block(32, 32)

        # Final Classifier
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def _make_up_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- Encoder Path ---
        skip_connections = []
        x = self.features(x)
        skip_connections.append(x)
        
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            if i != len(self.encoder_blocks) - 1:
                x = self.transition_blocks[i](x)
                skip_connections.append(x)
        
        # --- Bottleneck ---
        bottleneck = self.bottleneck_conv(x)

        # --- Decoder Path with Attention ---
        x = self.upsample1(bottleneck)
        skip1 = self.att1(skip_connections[5])
        skip1 = F.interpolate(skip1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.up_conv1(x)

        x = self.upsample2(x)
        skip2 = self.att2(skip_connections[3])
        skip2 = F.interpolate(skip2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.up_conv2(x)

        x = self.upsample3(x)
        skip3 = self.att3(skip_connections[1])
        skip3 = F.interpolate(skip3, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.up_conv3(x)
        
        x = self.upsample4(x)
        skip4 = self.att4(skip_connections[0])
        skip4 = F.interpolate(skip4, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip4], dim=1)
        x = self.up_conv4(x)
        
        x = self.upsample5(x)
        x = self.up_conv5(x)

        # --- Final Output ---
        out = self.classifier(x)
        return torch.sigmoid(out)

if __name__ == '__main__':
    # --- Unit Test for the Attention Model ---
    print("Running MicroSegNetAttention unit test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model
    model = MicroSegNetAttention(num_classes=1).to(device)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verification
    assert output.shape == dummy_input.shape, "Test Failed: Output shape does not match input shape."
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
    print("\n[SUCCESS] MicroSegNetAttention model test passed!")

