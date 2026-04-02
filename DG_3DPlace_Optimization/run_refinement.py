import torch
import cv2
import numpy as np
import os

# Import our modules
from src.refiner.gaussian_io import load_and_split_scene, merge_and_save_scene
from src.refiner.optimizer import PoseOptimizer
from src.utils.loss_utils import RefinementLoss
from src.utils.camera_utils import setup_camera_from_scout

# Import standard 3DGS rasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def run_refinement(ckpt_path, target_img_path, mask_path, num_object_gaussians, scout_camera_data, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Data Loading
    bg_gaussians, obj_gaussians = load_and_split_scene(ckpt_path, num_object_gaussians, device)
    
    target_rgb = torch.tensor(cv2.imread(target_img_path)[..., ::-1].copy()).permute(2,0,1).float().to(device) / 255.0
    target_mask = torch.tensor(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).copy()).unsqueeze(0).float().to(device) / 255.0
    
    camera = setup_camera_from_scout(scout_camera_data)
    
    # Background color for rasterization (usually black [0,0,0])
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    # 2. Setup Modules
    pose_model = PoseOptimizer(device=device)
    loss_module = RefinementLoss(device=device)
    
    optimizer = torch.optim.Adam([
        {'params': [pose_model.translation], 'lr': 0.005},
        {'params': [pose_model.rotation], 'lr': 0.001},
        {'params': [pose_model.scale], 'lr': 0.001}
    ])
    
    epochs = 200
    
    # 3. Main Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Apply pose
        transformed_obj = pose_model.transform_object(obj_gaussians)
        
        # Combine Gaussians for the forward pass
        combined_means = torch.cat([bg_gaussians['means'], transformed_obj['means']], dim=0)
        combined_scales = torch.cat([bg_gaussians['scales'], transformed_obj['scales']], dim=0)
        combined_rotations = torch.cat([bg_gaussians['rotations'], transformed_obj['rotations']], dim=0)
        combined_opacities = torch.cat([bg_gaussians['opacities'], transformed_obj['opacities']], dim=0)
        
        # Determine SH degree based on loaded features
        sh_degree = 3 if 'features_rest' in bg_gaussians else 0
        if 'features_dc' in bg_gaussians:
            combined_shs = torch.cat([bg_gaussians['features_dc'], transformed_obj['features_dc']], dim=0)
            if sh_degree > 0:
                combined_shs_rest = torch.cat([bg_gaussians['features_rest'], transformed_obj['features_rest']], dim=0)
                combined_shs = torch.cat([combined_shs, combined_shs_rest], dim=1)
        else:
            combined_colors = torch.cat([bg_gaussians['colors'], transformed_obj['colors']], dim=0)
            combined_shs = None

        # 4. Rasterization Settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=math.tan(camera.FoVx * 0.5),
            tanfovy=math.tan(camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # 5. Render
        rendered_image, radii = rasterizer(
            means3D=combined_means,
            means2D=torch.zeros_like(combined_means[:, :2], requires_grad=True, device=device),
            shs=combined_shs,
            colors_precomp=combined_colors if combined_shs is None else None,
            opacities=combined_opacities,
            scales=combined_scales,
            rotations=combined_rotations,
            cov3D_precomp=None
        )
        
        # 6. Mask Generation (Hack: Generate a 2D mask based on object indices)
        # We render a secondary image where object Gaussians are white and bg is black
        obj_colors = torch.ones((transformed_obj['means'].shape[0], 3), device=device)
        bg_colors = torch.zeros((bg_gaussians['means'].shape[0], 3), device=device)
        mask_colors = torch.cat([bg_colors, obj_colors], dim=0)
        
        rendered_mask_img, _ = rasterizer(
            means3D=combined_means.detach(), # Don't backprop through coordinates for the mask render
            means2D=torch.zeros_like(combined_means[:, :2], device=device),
            shs=None,
            colors_precomp=mask_colors,
            opacities=combined_opacities.detach(),
            scales=combined_scales.detach(),
            rotations=combined_rotations.detach(),
            cov3D_precomp=None
        )
        rendered_mask = rendered_mask_img[0:1, :, :] # Take one channel
        
        # 7. Compute Loss & Step
        loss, loss_dict = loss_module(rendered_image, target_rgb, rendered_mask, target_mask)
        
        loss.backward()
        optimizer.step()
        pose_model.normalize_quaternion()
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Total: {loss.item():.4f} | RGB: {loss_dict['rgb']:.4f} | LPIPS: {loss_dict['lpips']:.4f}")
            
    # 8. Save output
    print("[*] Optimization complete. Saving...")
    with torch.no_grad():
        final_obj = pose_model.transform_object(obj_gaussians)
        merge_and_save_scene(bg_gaussians, final_obj, output_path)

if __name__ == "__main__":
    import math
    import torch

    print("[*] Script started. Setting up mock camera...")
    
    # 1. Create a dummy camera just to let the rasterizer compile and run
    class MockCamera:
        def __init__(self):
            self.image_width = 512
            self.image_height = 512
            self.FoVx = math.pi / 3.0
            self.FoVy = math.pi / 3.0
            self.world_view_transform = torch.eye(4, device="cuda")
            self.full_proj_transform = torch.eye(4, device="cuda")
            self.camera_center = torch.tensor([0.0, 0.0, 0.0], device="cuda")
            
    dummy_camera = MockCamera()
    
    # 2. Execute the loop
    # IMPORTANT: Change 'num_object_gaussians' to roughly the number of 
    # points your 3D model has (e.g., if your .obj had 20,000 vertices, use 20000).
    try:
        run_refinement(
            ckpt_path="data/inputs/scene_with_initial_object.ckpt",
            target_img_path="data/inputs/diffusion_target.png",
            mask_path="data/inputs/object_mask.png",
            num_object_gaussians=20000,  # <-- Update this number!
            scout_camera_data=dummy_camera,
            output_path="data/outputs/scene_refined.ckpt"
        )
    except Exception as e:
        print(f"[!] An error occurred: {e}")