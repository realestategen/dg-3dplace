import sys
import torch
import os
import numpy as np
from PIL import Image
from rembg import remove

# Set memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# --- CONFIGURATION ---
INPUT_IMAGE_PATH = "input/demo1.png"
PROCESSED_IMAGE_PATH = "input/demo1_no_bg.png" # We save the clean image here
MESH_OUTPUT_PATH = "intermediate_mesh/mesh.obj"

print(f">>> [Step 1] Processing Image: {INPUT_IMAGE_PATH}")

# 1. REMOVE BACKGROUND
# We utilize the rembg library (installed with requirements)
try:
    input_img = Image.open(INPUT_IMAGE_PATH)
    
    print("    Removing background...")
    # 'remove' handles the alpha matting automatically
    processed_img = remove(input_img)
    
    # Save it so Step 2 can use exactly the same image
    processed_img.save(PROCESSED_IMAGE_PATH)
    print(f"    Clean image saved to: {PROCESSED_IMAGE_PATH}")

except Exception as e:
    print(f"Error processing image: {e}")
    exit()

print(">>> [Step 1] Loading Shape Model...")
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')

print(">>> [Step 1] Generating Mesh...")
# IMPORTANT: We use the PROCESSED image now, not the original
mesh_untextured = shape_pipeline(image=PROCESSED_IMAGE_PATH)[0]

# Save the mesh
mesh_untextured.export(MESH_OUTPUT_PATH)

print(f">>> [Step 1] Mesh saved to {MESH_OUTPUT_PATH}")
print(">>> [Step 1] Finished. Exiting to clear VRAM.")