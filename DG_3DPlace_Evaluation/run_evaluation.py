import os
import sys
import glob
from metrics import (
    clip_directional_similarity, 
    clip_text_directional_similarity, 
    dino_similarity, 
    calculate_ssim
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "2d_images")

# Paths for the Single-View (Image-to-Image) Metrics
PATH_INITIAL_2D = os.path.join(IMAGE_DIR, "initial_scene_render.png")
PATH_DIFFUSION_2D = os.path.join(IMAGE_DIR, "diffusion_guided.png")
PATH_FINAL_2D = os.path.join(IMAGE_DIR, "final_scene_render.png")

# Paths for the Multi-View (Text-to-Image) Metrics
# Create these folders and put your multi-angle renders inside them (e.g., view_01.png, view_02.png)
MULTI_VIEW_INITIAL_DIR = os.path.join(IMAGE_DIR, "multi_view", "initial")
MULTI_VIEW_FINAL_DIR = os.path.join(IMAGE_DIR, "multi_view", "final")

# Define your text prompts here
SOURCE_TEXT = "A home garden" # Describe the initial scene
TARGET_TEXT = "A home garden with a blue car" # Describe the scene with the new object

def evaluate_multi_view_text_clip():
    """Loops through multiple angles and averages the Text CLIP Directional Similarity."""
    print("\n--- Evaluating Multi-View Text CLIP Directional Similarity ---")
    
    if not os.path.exists(MULTI_VIEW_INITIAL_DIR) or not os.path.exists(MULTI_VIEW_FINAL_DIR):
        print(f"Skipping multi-view evaluation. Folders not found:\n{MULTI_VIEW_INITIAL_DIR}\n{MULTI_VIEW_FINAL_DIR}")
        return

    # Grab all image files in the initial directory
    initial_views = sorted(glob.glob(os.path.join(MULTI_VIEW_INITIAL_DIR, "*.png")))
    
    if not initial_views:
        print("No images found in the multi_view folders.")
        return

    scores = []
    for init_path in initial_views:
        filename = os.path.basename(init_path)
        final_path = os.path.join(MULTI_VIEW_FINAL_DIR, filename)
        
        if not os.path.exists(final_path):
            print(f"Warning: Missing matching final render for {filename}")
            continue

        try:
            score = clip_text_directional_similarity(
                init_path, final_path, SOURCE_TEXT, TARGET_TEXT
            )
            scores.append(score)
            print(f"View [{filename}]: {score:.4f}")
        except Exception as e:
            print(f"Failed on {filename}: {e}")

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f">>> Average Multi-View Text CLIP Score: {avg_score:.4f} <<<")


def evaluate_single_view_metrics():
    """Runs the original image-to-image metrics on the primary view."""
    print("\n--- Evaluating Single-View Metrics ---")
    required_files = [PATH_INITIAL_2D, PATH_DIFFUSION_2D, PATH_FINAL_2D]
    
    if not all(os.path.exists(f) for f in required_files):
        print("Skipping single-view metrics. Missing base images in data/2d_images/")
        return

    print("CLIP Directional (Image-to-Image):", f"{clip_directional_similarity(PATH_INITIAL_2D, PATH_FINAL_2D, PATH_DIFFUSION_2D):.4f}")
    print("DINOv2 Similarity:", f"{dino_similarity(PATH_FINAL_2D, PATH_DIFFUSION_2D):.4f}")
    print("Background SSIM:", f"{calculate_ssim(PATH_INITIAL_2D, PATH_FINAL_2D):.4f}")

if __name__ == "__main__":
    print("Starting DG-3DPlace Evaluation Pipeline...")
    evaluate_single_view_metrics()
    evaluate_multi_view_text_clip()
    print("\n--- Pipeline Complete ---")