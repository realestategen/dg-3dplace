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
        
    def forward(self, rendered_rgb, target_rgb, rendered_mask, target_mask, weights=(0.1, 0.1, 0.8)):
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
        
       # 3. Silhouette/Mask Loss (Standard BCE - flat gradient if no overlap)
        loss_mask = F.binary_cross_entropy(rendered_mask, target_mask)

        # --- ADD THIS NEW BLOCK ---
        # 4. Center of Mass Loss (Global gradient - pulls object from anywhere)
        H, W = rendered_mask.shape[1], rendered_mask.shape[2]
        y_coords = torch.arange(H, device=rendered_mask.device).float()
        x_coords = torch.arange(W, device=rendered_mask.device).float()
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Find the center of the rendered object
        rend_mass = rendered_mask.sum() + 1e-8
        rend_x = (rendered_mask[0] * grid_x).sum() / rend_mass
        rend_y = (rendered_mask[0] * grid_y).sum() / rend_mass

        # Find the center of the target area
        targ_mass = target_mask.sum() + 1e-8
        targ_x = (target_mask[0] * grid_x).sum() / targ_mass
        targ_y = (target_mask[0] * grid_y).sum() / targ_mass

        # Calculate the physical distance between the two centers
        loss_com = ((rend_x - targ_x)**2 + (rend_y - targ_y)**2) / (H * W)
        # --------------------------

        # Update total loss to include CoM (Weight it heavily so the object moves quickly)
        total_loss = (w_rgb * loss_rgb) + (w_lpips * loss_lpips) + (w_mask * loss_mask) + (5.0 * loss_com)
        
        return total_loss, {"rgb": loss_rgb.item(), "lpips": loss_lpips.item(), "mask": loss_mask.item()}