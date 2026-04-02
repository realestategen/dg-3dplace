import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class RefinementLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        # VGG is standard for LPIPS in view synthesis tasks
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        
    def forward(self, rendered_rgb, target_rgb, rendered_mask, target_mask, weights=(0.8, 0.1, 0.1)):
        """
        Computes the composite loss for refinement.
        Args:
            rendered_rgb: [3, H, W] tensor in [0, 1]
            target_rgb: [3, H, W] tensor in [0, 1]
            rendered_mask: [1, H, W] tensor in [0, 1]
            target_mask: [1, H, W] tensor in [0, 1]
            weights: Tuple of (RGB_weight, LPIPS_weight, Mask_weight)
        """
        w_rgb, w_lpips, w_mask = weights
        
        # 1. Photometric Loss (Masked to ignore background diffusion hallucinations)
        masked_rendered = rendered_rgb * target_mask
        masked_target = target_rgb * target_mask
        loss_rgb = self.l1_loss(masked_rendered, masked_target)
        
        # 2. Perceptual Loss (Requires inputs in [-1, 1] format)
        rendered_lpips = (rendered_rgb.unsqueeze(0) * 2) - 1
        target_lpips = (target_rgb.unsqueeze(0) * 2) - 1
        loss_lpips = self.lpips_fn(rendered_lpips, target_lpips).squeeze()
        
        # 3. Silhouette/Mask Loss
        loss_mask = F.binary_cross_entropy(rendered_mask, target_mask)
        
        total_loss = (w_rgb * loss_rgb) + (w_lpips * loss_lpips) + (w_mask * loss_mask)
        
        return total_loss, {"rgb": loss_rgb.item(), "lpips": loss_lpips.item(), "mask": loss_mask.item()}