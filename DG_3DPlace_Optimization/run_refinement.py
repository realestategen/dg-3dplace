import torch
import cv2
import math
import numpy as np
import os

# Import our modules
from src.refiner.gaussian_io import load_and_split_scene, merge_and_save_scene
from src.utils.loss_utils import RefinementLoss
from src.refiner.optimizer import PoseOptimizer
from src.utils.camera_utils import load_scout_camera

# Import standard 3DGS rasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch.nn as nn

def run_refinement(ckpt_path, target_img_path, mask_path, num_object_gaussians, scout_camera_data, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    bg_gaussians, obj_gaussians, full_ckpt = load_and_split_scene(ckpt_path, num_object_gaussians, device)
    
    if not os.path.exists(target_img_path) and os.path.exists(target_img_path.replace(".png", ".jpg")):
        target_img_path = target_img_path.replace(".png", ".jpg")
        
    target_rgb = torch.tensor(cv2.imread(target_img_path)[..., ::-1].copy()).permute(2,0,1).float().to(device) / 255.0
    target_mask = torch.tensor(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).copy()).unsqueeze(0).float().to(device) / 255.0
    
    camera = scout_camera_data
    camera.update_resolution(target_rgb.shape[2], target_rgb.shape[1])
    
    w2c = camera.world_view_transform.transpose(0, 1).clone()
    camera.world_view_transform = w2c.transpose(0, 1).contiguous()
    
    znear, zfar = 0.01, 100.0
    tanHalfFovY = math.tan(camera.FoVy * 0.5)
    tanHalfFovX = math.tan(camera.FoVx * 0.5)
    P = torch.zeros((4, 4), device=device)
    P[0, 0] = 1.0 / tanHalfFovX
    P[1, 1] = 1.0 / tanHalfFovY
    P[2, 2] = zfar / (zfar - znear)
    P[3, 2] = 1.0 
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    
    projmatrix = P.transpose(0, 1).contiguous()
    camera.full_proj_transform = torch.matmul(camera.world_view_transform, projmatrix).contiguous()
    camera.camera_center = torch.inverse(w2c)[0:3, 3].contiguous()

    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    # 2. Setup Modules
    pose_model = PoseOptimizer(device=device)
    loss_module = RefinementLoss(device=device)
    
    # ==========================================
    # [2] LOCK INITIAL POSITION (NO MORE TELEPORTING!)
    # ==========================================
    with torch.no_grad():
        # Find exactly where the blue blob already is in the .ckpt
        true_initial_center = obj_gaussians['means'].mean(dim=0)
        
        # Lock the optimizer's anchor and translation to that exact spot
        pose_model.obj_center = true_initial_center.clone()
        pose_model.translation.copy_(true_initial_center)
        
        print(f"[*] Object locked at its original .ckpt position: {true_initial_center}")
    # ==========================================

    optimizer = torch.optim.Adam([
        {'params': [pose_model.translation], 'lr': 0.02}, 
        {'params': [pose_model.rotation], 'lr': 0.01},
        {'params': [pose_model.scale_scalar], 'lr': 0.01} 
    ])

    epochs = 200
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        transformed_obj = pose_model.transform_object(obj_gaussians)
        
        combined_means = torch.cat([bg_gaussians['means'], transformed_obj['means']], dim=0).contiguous()
        combined_scales = torch.cat([bg_gaussians['scales'], transformed_obj['scales']], dim=0).contiguous()
        combined_rotations = torch.cat([bg_gaussians['rotations'], transformed_obj['rotations']], dim=0).contiguous()
        combined_opacities = torch.cat([bg_gaussians['opacities'], transformed_obj['opacities']], dim=0).contiguous()
        
        sh_degree = 3 if 'features_rest' in bg_gaussians else 0
        if 'features_dc' in bg_gaussians:
            combined_shs = torch.cat([bg_gaussians['features_dc'], transformed_obj['features_dc']], dim=0).contiguous()
            if combined_shs.dim() == 2:
                combined_shs = combined_shs.unsqueeze(1).contiguous()
            if sh_degree > 0:
                combined_shs_rest = torch.cat([bg_gaussians['features_rest'], transformed_obj['features_rest']], dim=0).contiguous()
                combined_shs = torch.cat([combined_shs, combined_shs_rest], dim=1).contiguous()
            combined_colors = None
        else:
            combined_colors = torch.cat([bg_gaussians['colors'], transformed_obj['colors']], dim=0).contiguous()
            combined_shs = None
            
        if combined_opacities.dim() == 1:
            combined_opacities = combined_opacities.unsqueeze(1).contiguous()

        active_scales = torch.exp(combined_scales).contiguous()
        active_opacities = torch.sigmoid(combined_opacities).contiguous()
        active_rotations = torch.nn.functional.normalize(combined_rotations, p=2, dim=-1).contiguous()

        obj_active_scales = torch.exp(transformed_obj['scales']).contiguous()
        obj_ops = transformed_obj['opacities']
        if obj_ops.dim() == 1:
            obj_ops = obj_ops.unsqueeze(1)
        obj_active_opacities = torch.sigmoid(obj_ops).contiguous()
        obj_active_rotations = torch.nn.functional.normalize(transformed_obj['rotations'], p=2, dim=-1).contiguous()

        rgb_raster_settings = GaussianRasterizationSettings(
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
        rgb_rasterizer = GaussianRasterizer(raster_settings=rgb_raster_settings)
        
        rendered_image, radii = rgb_rasterizer(
            means3D=combined_means,
            means2D=torch.zeros_like(combined_means, requires_grad=True, device=device),
            shs=combined_shs,
            colors_precomp=combined_colors,
            opacities=active_opacities,
            scales=active_scales,
            rotations=active_rotations,
            cov3D_precomp=None
        )
        
        # ==========================================
        # [4] MASK RASTERIZER (True Silhouette)
        # ==========================================
        mask_raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=math.tan(camera.FoVx * 0.5),
            tanfovy=math.tan(camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False
        )
        mask_rasterizer = GaussianRasterizer(raster_settings=mask_raster_settings)

        obj_colors = torch.ones((transformed_obj['means'].shape[0], 3), device=device).contiguous()
        
        # [!] REMOVED the opacity hack! We use the true, frozen opacities of the dragon.
        rendered_mask_img, _ = mask_rasterizer(
            means3D=transformed_obj['means'].contiguous(), 
            means2D=torch.zeros_like(transformed_obj['means'], requires_grad=True, device=device), 
            shs=None,
            colors_precomp=obj_colors,
            opacities=obj_active_opacities, # <--- True, soft opacities
            scales=obj_active_scales,
            rotations=obj_active_rotations,
            cov3D_precomp=None
        )
        # Apply a tiny threshold so we only see the solid parts of the dragon, not the fuzz
        rendered_mask = torch.clamp(rendered_mask_img[0:1, :, :], 0.0, 1.0)

        if epoch % 20 == 0:
            import torchvision
            os.makedirs("data/outputs/", exist_ok=True)
            torchvision.utils.save_image(rendered_image, f"data/outputs/DEBUG_camera_view_{epoch}.png")
            torchvision.utils.save_image(rendered_mask, f"data/outputs/DEBUG_rendered_mask_{epoch}.png")
        
        loss, loss_dict = loss_module(rendered_image, target_rgb, rendered_mask, target_mask)
        loss.backward()
        optimizer.step()
        pose_model.normalize_quaternion()
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Total: {loss.item():.4f} | MASK: {loss_dict['mask']:.4f} | RGB: {loss_dict['rgb']:.4f}")
            
    print("[*] Optimization complete. Saving...")
    with torch.no_grad():
        final_obj = pose_model.transform_object(obj_gaussians)
        merge_and_save_scene(bg_gaussians, final_obj, full_ckpt, output_path)

if __name__ == "__main__":
    try:
        real_camera = load_scout_camera("data/inputs/selected_camera.pt")
        run_refinement(
            ckpt_path="data/inputs/scene_with_initial_object.ckpt",
            target_img_path="data/inputs/diffusion_target.png",
            mask_path="data/inputs/object_mask.png",
            num_object_gaussians=15000, 
            scout_camera_data=real_camera, 
            output_path="data/outputs/scene_refined.ckpt"
        )
    except Exception as e:
        import traceback
        print(f"[!] An error occurred:\n{traceback.format_exc()}")