import torch
import json
import numpy as np
import yaml
from pathlib import Path
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig  # <--- NEW IMPORT

# ================= CONFIGURATION =================
# 1. Update this to the exact config path
config_path = Path("../room/output/my_scene/data/splatfacto/2026-02-02_124835/config.yml")

# 2. Local data path
local_data_path = Path("../room/data").resolve() 

output_dir = Path(".")
# =================================================

def load_config_and_patch_paths(config_path, new_data_path):
    """
    Loads config, patches paths, and fixes missing attributes from version mismatches.
    """
    print(f"Loading config from: {config_path}")
    
    # 1. Load the YAML manually
    config = yaml.load(config_path.open(), Loader=yaml.Loader)
    
    # 2. PATCH: Fix Paths
    print(f"⚠️  Patching data path: '{config.data}' -> '{new_data_path}'")
    config.data = new_data_path
    config.pipeline.datamanager.data = new_data_path
    
    # 3. PATCH: Fix Missing 'camera_optimizer' (The Fix for your Error)
    # The YAML loader sometimes skips defaults. We inject it manually if missing.
    if not hasattr(config.pipeline.model, "camera_optimizer"):
        print("⚠️  Patching missing 'camera_optimizer' attribute with default...")
        config.pipeline.model.camera_optimizer = CameraOptimizerConfig()

    # 4. Setup Pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="inference")
    
    # 5. Load Checkpoint
    ckpt_dir = config_path.parent / "nerfstudio_models"
    ckpt_path = sorted(ckpt_dir.glob("*.ckpt"))[-1]
    print(f"Loading checkpoint: {ckpt_path}")
    
    loaded_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    
    # Initialize missing attributes for inference mode
    pipeline.model.eval()
    
    # Fix missing attributes that splatfacto expects
    if not hasattr(pipeline.model, 'optimizers'):
        pipeline.model.optimizers = None
    if not hasattr(pipeline.model, 'strategy_state'):
        pipeline.model.strategy_state = None
    if not hasattr(pipeline.model, 'step'):
        pipeline.model.step = loaded_state["step"]
    if not hasattr(pipeline.model, 'info'):
        pipeline.model.info = {}
    
    return config, pipeline

def main():
    if not local_data_path.exists():
        raise FileNotFoundError(f"CRITICAL: Could not find data at {local_data_path}")

    # Load patched config
    config, pipeline = load_config_and_patch_paths(config_path, local_data_path)
    
    # Select Camera
    cameras = pipeline.datamanager.train_dataset.cameras
    camera_idx = 0 
    camera = cameras[camera_idx : camera_idx + 1]
    
    print(f"Rendering view from Camera Index {camera_idx}...")

    # Render
    outputs = pipeline.model.get_outputs_for_camera(camera)
    
    # Save RGB
    rgb = outputs["rgb"].cpu().numpy()
    rgb_img = Image.fromarray((rgb * 255).astype(np.uint8))
    rgb_path = output_dir / "background.png"
    rgb_img.save(rgb_path)
    print(f"✅ Saved Background Image: {rgb_path}")

    # Save Depth
    depth = outputs["depth"].cpu().numpy().squeeze()
    depth_path = output_dir / "depth.npy"
    np.save(depth_path, depth)
    print(f"✅ Saved Raw Depth Map: {depth_path}")

    # Save Metadata
    camera_meta = {
        "fx": float(camera.fx[0]),
        "fy": float(camera.fy[0]),
        "cx": float(camera.cx[0]),
        "cy": float(camera.cy[0]),
        "height": int(camera.height[0]),
        "width": int(camera.width[0]),
        "c2w": camera.camera_to_worlds[0].tolist()
    }
    
    meta_path = output_dir / "camera_meta.json"
    with open(meta_path, "w") as f:
        json.dump(camera_meta, f, indent=4)
    print(f"✅ Saved Camera Metadata: {meta_path}")

if __name__ == "__main__":
    main()