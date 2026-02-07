from rembg import remove
from PIL import Image

# Use the image created in Method 1
input_image = Image.open("output_images/output.png") 
output_image = remove(input_image)
output_image.save("output_images/chair_only_clean.png")