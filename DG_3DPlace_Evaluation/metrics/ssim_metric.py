from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_ssim(path_initial, path_final):
    # SSIM is typically computed on grayscale images
    img_initial = cv2.imread(path_initial, cv2.IMREAD_GRAYSCALE)
    img_final = cv2.imread(path_final, cv2.IMREAD_GRAYSCALE)
    
    # Ensure images are the same dimensions
    img_final = cv2.resize(img_final, (img_initial.shape[1], img_initial.shape[0]))
    
    # Calculate SSIM (score is between -1 and 1, where 1 is identical)
    score, diff_map = ssim(img_initial, img_final, full=True)
    return score