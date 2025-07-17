

# ---
# File: src/mamba_unet.py
# Description: Implementation of Mamba-UNet (HM-SegNet) based on the U-Mamba architecture.
# This model integrates Vision Mamba (VSS) blocks into a U-Net encoder.
# ---

import torch
import torch.nn as nn
from einops import rearrange

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None
    print("Warning: mamba-ssm is not installed. This model will not be usable.")

from .models_zoo.base_model.model import _Transition, _DenseBlock # Re-using components from MicroSegNet where applicable

# A basic Conv-Norm-Act block for residual connections
class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, norm_layer=nn.BatchNorm2d, act_layer=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# The core Vision State Space (VSS) Block as described in the technical report
class VSSBlock(nn.Module):
    def __init__(self, hidden_dim, drop_path=0., norm_layer=nn.LayerNorm, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.hidden_dim)
        self.d_conv = d_conv

        self.norm = norm_layer(hidden_dim)
        
        self.in_proj = nn.Linear(hidden_dim, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=self.d_conv, padding=(self.d_conv - 1) // 2, groups=self.d_inner, bias=False)
        
        self.act = nn.SiLU()

        # The Mamba module itself
        self.ssm = Mamba(
            d_model=self.d_inner,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=1, # The expansion is handled outside Mamba in this block
        )
        
        self.out_proj = nn.Linear(self.d_inner, hidden_dim, bias=False)

    def forward(self, x):
        # Input x has shape (B, H, W, C) for LayerNorm
        B, H, W, C = x.shape
        
        x_norm = self.norm(x)
        
        x_in, x_gate = self.in_proj(x_norm).chunk(2, dim=-1) # Split into two branches
        x_conv = self.conv2d(rearrange(x_in, 'b h w c -> b c h w')).permute(0, 2, 3, 1) # Apply DWConv
        x_conv_act = self.act(x_conv)

        # Apply Mamba (S6) block
        # Rearrange to (B, H*W, C) for Mamba
        x_mamba_in = rearrange(x_conv_act, 'b h w c -> b (h w) c')
        x_mamba_out = self.ssm(x_mamba_in)
        x_mamba_out = rearrange(x_mamba_out, 'b (h w) c -> b h w c', h=H, w=W)

        # Gating and output projection
        x_out = x_mamba_out * self.act(x_gate)
        x_out = self.out_proj(x_out)
        
        return x + x_out # Residual connection

# The U-Mamba Block, which combines Residual Conv blocks and a VSS block
class UMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, d_state=16):
        super().__init__()
        self.res_block1 = ConvNormAct(in_channels, out_channels, norm_layer=norm_layer)
        self.res_block2 = ConvNormAct(out_channels, out_channels, norm_layer=norm_layer)
        
        self.vss_block = VSSBlock(hidden_dim=out_channels, norm_layer=nn.LayerNorm, d_state=d_state)

    def forward(self, x):
        # 1. Pass through two residual conv blocks
        x_res = self.res_block1(x)
        x_res = self.res_block2(x_res)
        
        # 2. Pass through VSS block
        # VSSBlock expects (B, H, W, C)
        x_vss_in = x_res.permute(0, 2, 3, 1)
        x_vss_out = self.vss_block(x_vss_in)
        
        # Permute back to (B, C, H, W)
        x_vss_out = x_vss_out.permute(0, 3, 1, 2)
        
        # Add a residual connection
        return x_res + x_vss_out


class MambaUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, init_features=32, d_state=16):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba-ssm is not installed. Please install it to use MambaUNet.")

        features = init_features
        
        # --- Encoder ---
        # Start with standard convolutional blocks to extract robust local features
        self.encoder1 = nn.Sequential(
            ConvNormAct(in_channels, features),
            ConvNormAct(features, features)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = nn.Sequential(
            ConvNormAct(features, features * 2),
            ConvNormAct(features * 2, features * 2)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Introduce Mamba blocks in deeper, more semantic stages
        self.encoder3 = UMambaBlock(features * 2, features * 4, d_state=d_state)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = UMambaBlock(features * 4, features * 8, d_state=d_state)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = UMambaBlock(features * 8, features * 16, d_state=d_state)

        # --- Decoder (Convolutional for precise localization) ---
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = ConvNormAct(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvNormAct(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvNormAct(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = ConvNormAct(features * 2, features)

        # --- Final Classifier ---
        self.classifier = nn.Conv2d(features, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.classifier(dec1)


# --- Test Block ---
if __name__ == '__main__':
    if Mamba is None:
        print("Skipping MambaUNet test because mamba-ssm is not installed.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Testing MambaUNet on device: {device}")
        
        # Test with a dummy input
        model = MambaUNet(in_channels=1, num_classes=1).to(device)
        dummy_input = torch.randn(1, 1, 256, 256).to(device)
        
        try:
            output = model(dummy_input)
            print(f"Model Input Shape: {dummy_input.shape}")
            print(f"Model Output Shape: {output.shape}")
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
            print("\n[SUCCESS] MambaUNet test passed!")
            
        except Exception as e:
            print(f"\n[ERROR] An error occurred during the MambaUNet test: {e}")
            import traceback
            traceback.print_exc()


