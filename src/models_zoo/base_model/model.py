
import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class MicroSegNet(nn.Module):
    """
    MicroSegNet: A U-Net like architecture with DenseNet blocks in the encoder.
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):
        super(MicroSegNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # --- Encoder ---
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

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        
        # Bottleneck layer
        self.bottleneck_conv = nn.Conv2d(num_features, 512, kernel_size=1, stride=1, bias=False)
        
        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = self._make_up_conv_block(256 + encoder_channels[5], 256)

        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = self._make_up_conv_block(128 + encoder_channels[3], 128)

        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = self._make_up_conv_block(64 + encoder_channels[1], 64)
        
        self.upsample4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
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
        
        # Initial features
        x = self.features(x)
        skip_connections.append(x) # Skip 1
        
        # Dense Blocks & Transitions
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            if i != len(self.encoder_blocks) - 1:
                x = self.transition_blocks[i](x)
                skip_connections.append(x)
        
        # --- Bottleneck ---
        bottleneck = self.bottleneck_conv(x)

        # --- Decoder Path ---
        x = self.upsample1(bottleneck)
        skip = F.interpolate(skip_connections[5], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up_conv1(x)

        x = self.upsample2(x)
        skip = F.interpolate(skip_connections[3], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up_conv2(x)

        x = self.upsample3(x)
        skip = F.interpolate(skip_connections[1], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up_conv3(x)
        
        x = self.upsample4(x)
        skip = F.interpolate(skip_connections[0], size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.up_conv4(x)
        
        x = self.upsample5(x)
        x = self.up_conv5(x)

        # --- Final Output ---
        out = self.classifier(x)
        return torch.sigmoid(out) # Use sigmoid for binary segmentation

if __name__ == '__main__':
    # Test the model with a dummy input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MicroSegNet(num_classes=1).to(device)
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    output = model(dummy_input)
    print(f"Model Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")
    
    # Check the number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")

