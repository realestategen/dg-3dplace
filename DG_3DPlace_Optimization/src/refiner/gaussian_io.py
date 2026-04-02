import torch

def load_and_split_scene(ckpt_path, num_object_gaussians, device="cuda"):
    """
    Loads a .ckpt file and splits it into background and object Gaussians.
    Assumes the object Gaussians were appended to the end of the arrays.
    """
    # Adjust this loading mechanism based on how your 'Builder' saves the .ckpt
    full_scene = torch.load(ckpt_path, map_location=device)
    
    # Typically, a 3DGS checkpoint contains a dict of parameter tensors
    total_points = full_scene['means'].shape[0]
    
    if num_object_gaussians > total_points:
        raise ValueError("Object point count exceeds total scene point count.")
        
    split_idx = total_points - num_object_gaussians
    
    bg_gaussians = {}
    obj_gaussians = {}
    
    # Standard 3DGS parameters
    keys = ['means', 'rotations', 'scales', 'colors', 'opacities', 'features_dc', 'features_rest']
    
    for key in keys:
        if key in full_scene:
            bg_gaussians[key] = full_scene[key][:split_idx].detach()
            obj_gaussians[key] = full_scene[key][split_idx:].detach()
            
    return bg_gaussians, obj_gaussians

def merge_and_save_scene(bg_gaussians, optimized_obj_gaussians, output_path):
    """
    Combines the background and optimized object back into a single dictionary and saves it.
    """
    final_scene = {}
    for key in bg_gaussians.keys():
        final_scene[key] = torch.cat([bg_gaussians[key], optimized_obj_gaussians[key]], dim=0)
    
    torch.save(final_scene, output_path)