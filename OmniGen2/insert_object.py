import torch
import os
from PIL import Image
from diffusers.utils import load_image

# --- 1. IMPORT CHECK ---
try:
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you are running this from the folder where you cloned OmniGen2.")
    exit()

# --- CONFIGURATION ---
# Path to your empty room
BACKGROUND_PATH = "input_images/background.jpg" 
# Path to the object you want to insert (png or jpg)
OBJECT_PATH = "input_images/my_chair1.png"       
OUTPUT_PATH = "output_images/staged_room7.png"

MODEL_NAME = "OmniGen2/OmniGen2"

# --- PROMPT STRATEGY ---
# We explicitly mention both images using the special tags.
# "Edit the first image" tells the model the room is the base.
# "from the second image" tells it to use your specific object file.
# PROMPT = (
#     "<img><|image_1|></img> <img><|image_2|></img> "
#     "Edit the first image: Insert the armchair from the second image "
#     "into the center of the room. "
#     "The chair is standing upright on the floor, not flat. "
#     "It casts a realistic shadow on the ground. "
#     "Maintain the chair's 3D perspective and height."
# )

# PROMPT = (
#     "<img><|image_1|></img> <img><|image_2|></img> "
#     "Edit the first image: Insert the brown leather sofa from the second image "
#     "into the center of the room. "
#     "The sofa is standing upright on the floor, casting a realistic shadow. "
#     "Do not change the background walls, windows, or floor."
# )

# PROMPT = (
#     "<img><|image_1|></img> <img><|image_2|></img> "
#     "Edit the first image: Insert the sofa from the second image "
#     "into coordinates [150, 550, 850, 900]. "
#     "The sofa is wide and has 3 seats. "
#     "Ensure it blends with the floor shadows but keeps the background unchanged."
# )

PROMPT = (
    "<img><|image_1|></img> <img><|image_2|></img> "
    "Add the three seats sofa from image 2 onto the floor in image 1 "
    "into coordinates [250, 550, 750, 850]. "
    "The sofa is placed naturally on the wooden floor with a realistic shadow. "
    "Keep the background walls, window, and floor texture exactly unchanged."
)


def main():
    print(f"--- Starting Virtual Staging with {MODEL_NAME} ---")
    
    # 1. Load Pipeline
    pipe = OmniGen2Pipeline.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 2. Fixes & Optimizations
    # Fix for 'enable_teacache' error if it occurs in your version
    if hasattr(pipe, 'transformer'):
         pipe.transformer.enable_teacache = False
    
    # Enable CPU Offload to save VRAM (Crucial for 16GB cards)
    pipe.enable_model_cpu_offload()
    
    # 3. Load Images
    if not os.path.exists(BACKGROUND_PATH) or not os.path.exists(OBJECT_PATH):
        print(f"Error: Missing input files.\nCheck: {BACKGROUND_PATH}\nCheck: {OBJECT_PATH}")
        return

    img_room = load_image(BACKGROUND_PATH)
    img_object = load_image(OBJECT_PATH)
    
    # Resize logic: Align background to multiples of 16 (Required by VAE)
    w, h = img_room.size
    w = w - (w % 16)
    h = h - (h % 16)
    
    # Optional: Resize object image if it's massive (saves RAM)
    # img_object.thumbnail((1024, 1024))

    print(f"Room Size: {w}x{h}")
    print(f"Object Size: {img_object.size}")
    print(f"Prompt: {PROMPT}")

    # 4. Generate
    generator = torch.Generator(device="cpu").manual_seed(42)

    result = pipe(
        prompt=PROMPT,
        input_images=[img_room, img_object],   # <--- LIST OF BOTH IMAGES
        height=h,
        width=w,
        text_guidance_scale=2.5,       # How much it listens to text
        image_guidance_scale=2.2,   # <--- HIGH VALUE (2.5-3.0) forces it to use YOUR object, not a random one
        generator=generator
    )

    # 5. Save
    output = result.images[0]
    output.save(OUTPUT_PATH)
    print(f"Success! Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()