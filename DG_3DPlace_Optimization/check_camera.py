import torch
from src.utils.camera_utils import load_scout_camera

print("\n" + "="*50)
print("--- LOADING REAL CAMERA ---")
try:
    real_camera = load_scout_camera("data/inputs/selected_camera.pt")
    
    print(f"REAL Camera Center (XYZ): {real_camera.camera_center}")
    print(f"REAL World View Transform:\n{real_camera.world_view_transform}")
    
    checkpoint = torch.load("data/inputs/scene_with_initial_object.ckpt", map_location="cuda")
    means = checkpoint[0]['params']['xyz'] if 'params' in checkpoint[0] else checkpoint[0]['means']
    
    print("\n--- SCENE BOUNDS ---")
    print(f"Scene Min Bounds: {means.min(dim=0)[0]}")
    print(f"Scene Max Bounds: {means.max(dim=0)[0]}")
    print("="*50 + "\n")
except Exception as e:
    print(f"[!] Error loading camera: {e}")