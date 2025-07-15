

import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer
import numpy as np
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A basic convolutional block with two conv layers."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class DecoderCup(nn.Module):
    """The decoder part of the TransUNet, responsible for upsampling."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = ConvBlock(config.hidden_size, head_channels)
        
        # Upsampling blocks
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBlock(head_channels, 256)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = ConvBlock(256, 128)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = ConvBlock(128, 64)
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = ConvBlock(64, 16)

    def forward(self, hidden_states):
        # Remove the CLS token (first token) and reshape the transformer output to a 2D feature map
        B, seq_len, hidden = hidden_states.shape
        # Remove CLS token (first token)
        patch_embeddings = hidden_states[:, 1:, :]  # Skip the first token (CLS token)
        n_patches = patch_embeddings.shape[1]
        h, w = int(np.sqrt(n_patches)), int(np.sqrt(n_patches))
        x = patch_embeddings.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        
        x = self.conv_more(x)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)
        return x

class TransUNet(nn.Module):
    """
    The TransUNet architecture. It uses a Vision Transformer as an encoder
    and a CNN-based decoder for upsampling.
    """
    def __init__(self, *, img_size=256, num_classes=1, vit_model_name='vit_base_patch16_224_in21k', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # --- Vision Transformer Encoder ---
        # Load a pre-trained Vision Transformer from timm
        # We need to adapt it for single-channel input
        self.vit = timm.create_model(vit_model_name, pretrained=pretrained)
        
        # Modify the first convolution layer to accept 1 channel instead of 3
        original_pos_embed = self.vit.pos_embed
        original_patch_embed = self.vit.patch_embed.proj
        
        self.vit.patch_embed.proj = nn.Conv2d(1, original_patch_embed.out_channels,
                                              kernel_size=original_patch_embed.kernel_size,
                                              stride=original_patch_embed.stride,
                                              padding=original_patch_embed.padding,
                                              bias=original_patch_embed.bias is not None)
        # The original ViT was trained on 3 channels, so we average the weights for our new 1-channel conv
        with torch.no_grad():
            self.vit.patch_embed.proj.weight.copy_(original_patch_embed.weight.mean(1, keepdim=True))

        # Update the image size in the patch embedding module
        if isinstance(self.vit.patch_embed, timm.layers.PatchEmbed):
            self.vit.patch_embed.img_size = (img_size, img_size)

        # Adjust position embedding if image size is different from pre-training
        if img_size != 224:
            self.vit.pos_embed = nn.Parameter(self.resize_pos_embed(original_pos_embed, (img_size, img_size)))

        # --- Decoder ---
        # Create a dummy config object for the decoder
        class DummyConfig:
            hidden_size = self.vit.embed_dim
        
        self.decoder = DecoderCup(DummyConfig())
        
        # --- Final Output Layer ---
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Use forward_features to get the patch embeddings before the classification head
        hidden_states = self.vit.forward_features(x)
        
        # Pass features to the decoder
        decoder_output = self.decoder(hidden_states)
        
        # Get final segmentation map
        logits = self.segmentation_head(decoder_output)
        
        return torch.sigmoid(logits)

    @staticmethod
    def resize_pos_embed(posemb, new_shape):
        """Resize position embeddings for different image sizes."""
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        gs_old = int(np.sqrt(len(posemb_grid)))
        gs_new = (new_shape[0] // 16, new_shape[1] // 16)
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb

if __name__ == '__main__':
    print("Running TransUNet unit test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model (use a smaller ViT for faster testing if needed)
    # Note: Set pretrained=False if you don't have an internet connection during test
    model = TransUNet(img_size=256, num_classes=1, vit_model_name='vit_tiny_patch16_224', pretrained=False).to(device)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verification
    assert output.shape == (2, 1, 256, 256), "Test Failed: Output shape is incorrect."
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
    print("\n[SUCCESS] TransUNet model test passed!")

