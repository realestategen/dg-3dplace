import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 1. Define the device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load DINOv2 and move it to the GPU
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.to(device)
dinov2.eval()

# Transform pipeline for DINO
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def dino_similarity(path_final, path_diffusion):
    # 3. Load images and move them to the GPU
    img_final = transform(Image.open(path_final).convert("RGB")).unsqueeze(0).to(device)
    img_diff = transform(Image.open(path_diffusion).convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat_final = dinov2(img_final)
        feat_diff = dinov2(img_diff)
    
    # Normalize and compute similarity
    feat_final = F.normalize(feat_final, p=2, dim=-1)
    feat_diff = F.normalize(feat_diff, p=2, dim=-1)
    
    similarity = F.cosine_similarity(feat_final, feat_diff)
    return similarity.item()