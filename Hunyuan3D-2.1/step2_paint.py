import sys
import torch
import os
import gc
import shutil

# 1. Set memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# --- CONFIGURATION ---
OUTPUT_DIR = "output"

# CORRECTED INPUT PATH
MESH_INPUT = "intermediate_mesh/mesh.obj" 
IMAGE_INPUT = "input/demo1_no_bg.png"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(">>> [Step 2] Loading Paint Model...")
paint_config = Hunyuan3DPaintConfig(max_num_view=9, resolution=512)
paint_pipeline = Hunyuan3DPaintPipeline(paint_config)

print(">>> [Step 2] Texturing Mesh...")
if not os.path.exists(MESH_INPUT):
    print(f"Error: {MESH_INPUT} not found! Did Step 1 run successfully?")
    exit()

# Clear memory
gc.collect()
torch.cuda.empty_cache()

# --- RUN THE PIPELINE ---
# The pipeline saves the textured files into the SAME folder as the input mesh
_ = paint_pipeline(MESH_INPUT, image_path=IMAGE_INPUT)

# --- MOVE LOGIC ---
# The pipeline puts output files in the folder where the input mesh was.
source_folder = os.path.dirname(MESH_INPUT) 
print(f"\n>>> Scanning '{source_folder}' for generated files...")

if os.path.exists(source_folder):
    files_moved = 0
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        
        # We want to move the TEXTURED results (usually mesh_textured.obj or .glb)
        # We avoid moving the original input "mesh.obj" unless you want it.
        if filename.lower().endswith(('.mtl', '.png', '.jpg', '.jpeg', '.glb')):
            destination_path = os.path.join(OUTPUT_DIR, filename)
            shutil.move(source_path, destination_path)
            print(f"    -> Moved: {filename}")
            files_moved += 1
        
        # Handle the textured OBJ specifically (sometimes named differently)
        elif filename.endswith(".obj") and filename != "mesh.obj":
             destination_path = os.path.join(OUTPUT_DIR, filename)
             shutil.move(source_path, destination_path)
             print(f"    -> Moved: {filename}")
             files_moved += 1

    if files_moved > 0:
        print("-" * 40)
        print(f">>> SUCCESS! {files_moved} files saved to '{OUTPUT_DIR}/'")
        print("-" * 40)
    else:
        print(f"!!! WARNING: No new model files found in {source_folder}")
else:
    print(f"!!! ERROR: The folder '{source_folder}' does not exist.")