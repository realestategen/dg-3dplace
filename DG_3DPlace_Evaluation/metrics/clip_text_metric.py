import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load model and processor
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

def get_clip_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # BULLETPROOF FIX: Provide dummy text 
    inputs = processor(text=["dummy text"], images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.image_embeds
        
    return embedding / embedding.norm(p=2, dim=-1, keepdim=True)

def get_clip_text_embedding(text):
    # BULLETPROOF FIX: Provide a dummy image (a blank black square)
    dummy_image = Image.new("RGB", (224, 224), (0, 0, 0))
    
    inputs = processor(text=[text], images=dummy_image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.text_embeds
        
    return embedding / embedding.norm(p=2, dim=-1, keepdim=True)

def clip_text_directional_similarity(path_initial, path_final, source_text, target_text):
    """
    Measures if the change in the images matches the change in the text prompts.
    """
    # 1. Get image direction vector
    emb_img_initial = get_clip_image_embedding(path_initial)
    emb_img_final = get_clip_image_embedding(path_final)
    dir_image = emb_img_final - emb_img_initial

    # 2. Get text direction vector
    emb_txt_initial = get_clip_text_embedding(source_text)
    emb_txt_final = get_clip_text_embedding(target_text)
    dir_text = emb_txt_final - emb_txt_initial

    # 3. Calculate alignment
    similarity = F.cosine_similarity(dir_image, dir_text)
    return similarity.item()