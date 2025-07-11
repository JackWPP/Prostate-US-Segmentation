

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM)
    This module learns to assign different weights to each channel,
    effectively selecting which features are more important.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP for both average and max pooled features
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        # Element-wise summation, followed by sigmoid activation
        attention_weights = self.sigmoid(avg_out + max_out)
        return x * attention_weights

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM)
    This module learns to focus on specific regions of the feature map,
    identifying "where" the important information is.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Ensure padding is 3 for a 7x7 kernel to keep feature map size constant
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average and max pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate the pooled features
        concatenated = torch.cat([avg_out, max_out], dim=1)
        
        # Generate the spatial attention map
        attention_map = self.sigmoid(self.conv(concatenated))
        return x * attention_map

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines Channel Attention and Spatial Attention sequentially.
    This allows the model to learn "what" and "where" to focus.
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # First, apply channel attention
        x = self.channel_attention(x)
        # Then, apply spatial attention
        x = self.spatial_attention(x)
        return x

if __name__ == '__main__':
    # --- Unit Test for CBAM ---
    # This test ensures the module works as a standalone component.
    print("Running CBAM unit test...")
    
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    dummy_input = torch.randn(4, 64, 32, 32)
    print(f"Input tensor shape: {dummy_input.shape}")

    # Initialize the CBAM module
    cbam_module = CBAM(in_planes=64)
    
    # Pass the dummy input through the module
    output = cbam_module(dummy_input)
    
    print(f"Output tensor shape: {output.shape}")

    # Verification
    assert output.shape == dummy_input.shape, "Test Failed: Output shape does not match input shape."
    
    print("\n[SUCCESS] CBAM module test passed!")
    print("The output shape is identical to the input shape, as expected.")

