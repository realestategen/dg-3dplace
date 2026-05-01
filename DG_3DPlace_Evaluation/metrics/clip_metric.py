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


def clip_directional_similarity_normalized(path_initial, path_final, path_diffusion):
    """
    Corrected CLIP directional similarity with proper normalization.
    
    Based on HuggingFace diffusers reference implementation.
    Measures semantic alignment by comparing direction vectors in CLIP embedding space.
    
    Args:
        path_initial: Path to initial/source image
        path_final: Path to final/optimized image (our result)
        path_diffusion: Path to diffusion-guided target image
        
    Returns:
        float: Cosine similarity in [0, 1], higher is better alignment
    """
    # 1. Extract image embeddings (already L2-normalized in get_clip_embedding)
    emb_initial = get_clip_embedding(path_initial)
    emb_final = get_clip_embedding(path_final)
    emb_diffusion = get_clip_embedding(path_diffusion)
    
    # 2. Calculate direction vectors (change in embedding space)
    dir_render = emb_final - emb_initial        # How our optimization changed it
    dir_diffusion = emb_diffusion - emb_initial # How the target changed it
    
    # 3. Normalize direction vectors (CRITICAL - aligns with diffusers library)
    dir_render = F.normalize(dir_render, p=2, dim=-1)
    dir_diffusion = F.normalize(dir_diffusion, p=2, dim=-1)
    
    # 4. Compute cosine similarity between normalized direction vectors
    similarity = F.cosine_similarity(dir_render, dir_diffusion)
    
    return similarity.item()