

# ---
# File: src/unet_refiner.py
# Description: Implements a UNet + Mamba Refiner architecture.
# This model first uses a standard UNet to generate a segmentation map,
# then uses a Mamba-based block to refine that map.
# ---

import torch
import torch.nn as nn
from einops import rearrange

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    Mamba = None

# Import the standard UNet
from .models_zoo.unet.model import UNet

# --- VSSBlock (from previous attempts) ---
class VSSBlock(nn.Module):
    def __init__(self, hidden_dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_inner = int(expand * hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv, 
            padding=(d_conv - 1) // 2, groups=self.d_inner, bias=False
        )
        self.act = nn.SiLU()
        self.ssm = Mamba(d_model=self.d_inner, d_state=d_state, d_conv=d_conv, expand=1)
        self.out_proj = nn.Linear(self.d_inner, hidden_dim, bias=False)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 3, 1)
        B, H, W, C = x_permuted.shape
        x_in, x_gate = self.in_proj(x_permuted).chunk(2, dim=-1)
        x_conv = self.conv2d(x_in.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_conv_act = self.act(x_conv)
        x_mamba_in = rearrange(x_conv_act, 'b h w c -> b (h w) c')
        x_mamba_out = self.ssm(x_mamba_in)
        x_mamba_out = rearrange(x_mamba_out, 'b (h w) c -> b h w c', h=H, w=W)
        x_out = x_mamba_out * self.act(x_gate)
        x_out = self.out_proj(x_out)
        return x_out.permute(0, 3, 1, 2)

# --- MambaRefinerBlock ---
# A shallow network to refine the UNet's output.
class MambaRefiner(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_blocks=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, bias=False)
        
        self.blocks = nn.Sequential(*[
            VSSBlock(hidden_dim=embed_dim) for _ in range(num_blocks)
        ])
        
        self.out_conv = nn.Conv2d(embed_dim, in_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        # x is the output from the UNet (a 1-channel mask)
        x = self.in_conv(x)
        x = self.blocks(x)
        x = self.out_conv(x)
        return x

# --- UNetWithMambaRefiner ---
class UNetWithMambaRefiner(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, use_sigmoid=True):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba-ssm is not installed. Please install it to use this model.")
            
        self.use_sigmoid = use_sigmoid
        
        # The UNet part does not have a sigmoid, as we'll apply it at the very end.
        self.unet = UNet(n_channels=n_channels, n_classes=n_classes)
        # We need to modify the UNet to output logits instead of probabilities
        self.unet.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        # The Mamba part refines the UNet's output
        self.refiner = MambaRefiner(in_channels=n_classes)

    def forward(self, x):
        # 1. Get initial segmentation from the standard UNet
        unet_logits = self.unet(x)
        
        # 2. Refine the logits with the Mamba block
        refined_logits = self.refiner(unet_logits)
        
        # The final output is the sum of the original and the refinement,
        # acting as a residual connection.
        final_logits = unet_logits + refined_logits
        
        if self.use_sigmoid:
            return torch.sigmoid(final_logits)
        return final_logits

# --- Test Block ---
if __name__ == '__main__':
    if Mamba is None:
        print("Skipping UNetWithMambaRefiner test because mamba-ssm is not installed.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Testing UNetWithMambaRefiner on device: {device}")
        
        model = UNetWithMambaRefiner(n_channels=1, n_classes=1).to(device)
        dummy_input = torch.randn(2, 1, 256, 256).to(device)
        
        try:
            output = model(dummy_input)
            print(f"Model Input Shape: {dummy_input.shape}")
            print(f"Model Output Shape: {output.shape}")
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
            print("\n[SUCCESS] UNetWithMambaRefiner test passed!")
            
        except Exception as e:
            print(f"\n[ERROR] An error occurred during the test: {e}")
            import traceback
            traceback.print_exc()

