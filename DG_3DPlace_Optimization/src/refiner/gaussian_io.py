import torch

def load_and_split_scene(ckpt_path, num_object_gaussians=None, device="cuda"):
    full_ckpt = torch.load(ckpt_path, map_location=device)
    
    # ==========================================
    # DYNAMIC OBJECT COUNT DETECTION
    # ==========================================
    if num_object_gaussians is None:
        # Strategy 1: Check for explicitly saved metadata
        if 'num_object_gaussians' in full_ckpt:
            num_object_gaussians = full_ckpt['num_object_gaussians']
            print(f"[*] IO Success: Found metadata num_object_gaussians = {num_object_gaussians}")
            
        # Strategy 2: Deduce from the features_rest tensor mismatch
        elif 'pipeline' in full_ckpt and '_model.features_rest' in full_ckpt['pipeline']:
            total_splats = full_ckpt['pipeline']['_model.means'].shape[0]
            bg_splats = full_ckpt['pipeline']['_model.features_rest'].shape[0]
            
            if total_splats > bg_splats:
                num_object_gaussians = total_splats - bg_splats
                print(f"[*] IO Success: Calculated from tensor shapes num_object_gaussians = {num_object_gaussians}")
                
        # Fallback
        if num_object_gaussians is None or num_object_gaussians <= 0:
            print("[!] IO Warning: Could not dynamically detect object count. Defaulting to 15000.")
            num_object_gaussians = 15000
    # ==========================================

    pipeline_state = full_ckpt['pipeline']
    total_points = pipeline_state['_model.means'].shape[0]
    split_idx = total_points - num_object_gaussians
    
    bg_gaussians = {}
    obj_gaussians = {}
    
    key_map = {
        'means': '_model.means',
        'scales': '_model.scales',
        'rotations': '_model.quats',
        'opacities': '_model.opacities',
        'features_dc': '_model.features_dc',
        'features_rest': '_model.features_rest'
    }
    
    for internal_key, ckpt_key in key_map.items():
        if ckpt_key in pipeline_state:
            tensor = pipeline_state[ckpt_key]
            
            # Normal case: Builder added points to this array
            if tensor.shape[0] == total_points:
                bg_gaussians[internal_key] = tensor[:split_idx].detach()
                obj_gaussians[internal_key] = tensor[split_idx:].detach()
                
            # Edge case: Builder skipped this array (e.g. features_rest)
            elif tensor.shape[0] == total_points - num_object_gaussians:
                print(f"[*] IO Note: Generating missing {internal_key} for object.")
                bg_gaussians[internal_key] = tensor.detach()
                
                # Create an array of zeros matching the expected shape
                obj_shape = (num_object_gaussians,) + tensor.shape[1:]
                obj_gaussians[internal_key] = torch.zeros(obj_shape, dtype=tensor.dtype, device=device)
            else:
                print(f"[!] IO Warning: {ckpt_key} has unexpected shape {tensor.shape}")
                
    return bg_gaussians, obj_gaussians, full_ckpt

def merge_and_save_scene(bg_gaussians, optimized_obj_gaussians, full_ckpt, output_path):
    key_map = {
        'means': '_model.means',
        'scales': '_model.scales',
        'rotations': '_model.quats',
        'opacities': '_model.opacities',
        'features_dc': '_model.features_dc',
        'features_rest': '_model.features_rest'
    }
    
    # Calculate what the total points should be
    total_points = bg_gaussians['means'].shape[0] + optimized_obj_gaussians['means'].shape[0]
    
    for internal_key, ckpt_key in key_map.items():
        if internal_key in bg_gaussians and ckpt_key in full_ckpt['pipeline']:
            original_tensor = full_ckpt['pipeline'][ckpt_key]
            
            # Only overwrite if the Builder actually originally saved data here
            if original_tensor.shape[0] == total_points:
                merged_tensor = torch.cat([bg_gaussians[internal_key], optimized_obj_gaussians[internal_key]], dim=0)
                full_ckpt['pipeline'][ckpt_key] = merged_tensor
                
    torch.save(full_ckpt, output_path)
    print(f"[*] Refined scene safely saved to {output_path}")