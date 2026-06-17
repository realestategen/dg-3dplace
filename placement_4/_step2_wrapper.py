import sys
import os

# Change to Hunyuan3D-2.1 directory so relative imports work
os.chdir(r'/home/cse_g2/RealEstateGen/DG-3DPlace/Hunyuan3D-2.1')
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# Set input/output paths
IMAGE_INPUT = r'/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260618_031908/gemini_object_cutout.png'
MESH_INPUT = r'/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260618_031908/generated_object.obj'
OUTPUT_DIR = r'/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260618_031908'

# Execute step2_paint.py code
with open(r'/home/cse_g2/RealEstateGen/DG-3DPlace/Hunyuan3D-2.1/step2_paint.py') as code_file:
    code = code_file.read()
    # Replace the default paths with our absolute paths
    code = code.replace('IMAGE_INPUT = "input/demo_no_bg.png"',
                       f'IMAGE_INPUT = r"/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260618_031908/gemini_object_cutout.png"')
    code = code.replace('MESH_INPUT = "intermediate_mesh/mesh.obj"',
                       f'MESH_INPUT = r"/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260618_031908/generated_object.obj"')
    code = code.replace('OUTPUT_DIR = "output"',
                       f'OUTPUT_DIR = r"/home/cse_g2/RealEstateGen/DG-3DPlace/placement_4/session_20260618_031908"')
    code = code.replace(
        'paint_config = Hunyuan3DPaintConfig(max_num_view=9, resolution=512)',
        'paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=384)'
    )
    exec(code)
