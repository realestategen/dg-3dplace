import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load model and processor
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

def get_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # BULLETPROOF FIX: Provide dummy text so the main forward pass succeeds
    inputs = processor(text=["dummy text"], images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # This is now guaranteed to be a tensor
        embedding = outputs.image_embeds
        
    return embedding / embedding.norm(p=2, dim=-1, keepdim=True)

def clip_directional_similarity(path_initial, path_final, path_diffusion):
    emb_initial = get_clip_embedding(path_initial)
    emb_final = get_clip_embedding(path_final)
    emb_diffusion = get_clip_embedding(path_diffusion)

    # Calculate direction vectors
    dir_render = emb_final - emb_initial
    dir_diffusion = emb_diffusion - emb_initial

    # Cosine similarity
    similarity = F.cosine_similarity(dir_render, dir_diffusion)
    return similarity.item()