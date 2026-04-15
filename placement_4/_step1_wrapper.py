import sys
import os

# Change to Hunyuan3D-2.1 directory so relative imports work
os.chdir(r'/home/cse_g2/RealEstateGen/DG-3DPlace/Hunyuan3D-2.1')
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Set input/output paths
INPUT_IMAGE_PATH = r'/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260416_022532/gemini_object_cutout.png'
PROCESSED_IMAGE_PATH = r'/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260416_022532/gemini_object_cutout_processed.png'
MESH_OUTPUT_PATH = r'/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260416_022532/generated_object.obj'

# Execute step1_shape.py code
with open(r'/home/cse_g2/RealEstateGen/DG-3DPlace/Hunyuan3D-2.1/step1_shape.py') as code_file:
    code = code_file.read()
    # Replace the default paths with our absolute paths
    code = code.replace('INPUT_IMAGE_PATH = "input/demo.png"', 
                       f'INPUT_IMAGE_PATH = r"/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260416_022532/gemini_object_cutout.png"')
    code = code.replace('PROCESSED_IMAGE_PATH = "input/demo_no_bg.png"',
                       f'PROCESSED_IMAGE_PATH = r"/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260416_022532/gemini_object_cutout_processed.png"')
    code = code.replace('MESH_OUTPUT_PATH = "intermediate_mesh/mesh.obj"',
                       f'MESH_OUTPUT_PATH = r"/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260416_022532/generated_object.obj"')
    exec(code)
