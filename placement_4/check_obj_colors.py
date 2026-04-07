#!/usr/bin/env python3
"""
Simple diagnostic script to check if OBJ files have colors/textures applied.
Helps debug why generated objects appear white/colorless.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import trimesh
    import numpy as np
except ImportError:
    print("Error: trimesh or numpy not found. Install with: pip install trimesh numpy")
    sys.exit(1)


def check_obj_colors(obj_path):
    """Load OBJ and report color/texture information."""
    
    if not os.path.exists(obj_path):
        print(f"[ERROR] File not found: {obj_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Checking OBJ file: {obj_path}")
    print(f"{'='*60}\n")
    
    try:
        # Load mesh
        mesh = trimesh.load(obj_path, process=False)
        print(f"✓ Mesh loaded successfully")
        print(f"  - Vertices: {len(mesh.vertices)}")
        print(f"  - Faces: {len(mesh.faces)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load mesh: {e}")
        return False
    
    # Check for vertex colors
    print(f"\n[1] Vertex Colors:")
    if hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
        vc = mesh.vertex_colors
        if len(vc) > 0:
            print(f"  ✓ Found {len(vc)} vertex colors")
            print(f"    Sample colors (first 3): {vc[:3]}")
        else:
            print(f"  ✗ Vertex colors array is empty")
    else:
        print(f"  ✗ No vertex colors attribute")
    
    # Check for materials
    print(f"\n[2] Materials:")
    if hasattr(mesh, 'visual'):
        visual = mesh.visual
        print(f"  Visual type: {type(visual).__name__}")
        
        if hasattr(visual, 'material'):
            print(f"  ✓ Material found: {visual.material}")
        else:
            print(f"  ✗ No material attribute")
    else:
        print(f"  ✗ No visual attribute")
    
    # Check for face colors
    print(f"\n[3] Face Colors:")
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colors'):
        fc = mesh.visual.face_colors
        if fc is not None and len(fc) > 0:
            print(f"  ✓ Found {len(fc)} face colors")
            print(f"    Sample colors (first 3): {fc[:3]}")
            # Check if all colors are identical (likely white/default)
            unique_colors = np.unique(fc, axis=0)
            if len(unique_colors) == 1:
                color = unique_colors[0]
                print(f"  ⚠ All faces have same color: RGBA{tuple(color)}")
                if np.allclose(color[:3], [255, 255, 255]) or np.allclose(color[:3], [1, 1, 1]):
                    print(f"    → ISSUE: Mesh is entirely WHITE (no texture applied)")
            else:
                print(f"  ✓ Multiple colors found ({len(unique_colors)} unique colors)")
        else:
            print(f"  ✗ No face colors")
    else:
        print(f"  ✗ Cannot access face colors")
    
    # Check for texture files
    print(f"\n[4] Texture Files:")
    obj_dir = os.path.dirname(obj_path)
    obj_base = os.path.splitext(os.path.basename(obj_path))[0]
    
    texture_patterns = [
        f"{obj_base}.jpg",
        f"{obj_base}.png",
        f"{obj_base}_albedo.jpg",
        f"{obj_base}_albedo.png",
        f"{obj_base}.mtl",
        "albedo.jpg",
        "albedo.png",
    ]
    
    found_textures = []
    for pattern in texture_patterns:
        texture_path = os.path.join(obj_dir, pattern)
        if os.path.exists(texture_path):
            size_mb = os.path.getsize(texture_path) / (1024 * 1024)
            found_textures.append(f"{pattern} ({size_mb:.1f} MB)")
    
    if found_textures:
        print(f"  ✓ Found texture files:")
        for t in found_textures:
            print(f"    - {t}")
    else:
        print(f"  ✗ No texture files found in {obj_dir}")
    
    # Check MTL file
    print(f"\n[5] MTL Material File:")
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    if os.path.exists(mtl_path):
        print(f"  ✓ MTL file found: {mtl_path}")
        # Read and print first few lines
        with open(mtl_path, 'r') as f:
            lines = f.readlines()[:15]
            for line in lines:
                print(f"    {line.rstrip()}")
    else:
        print(f"  ✗ No MTL file found")
    
    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSIS:")
    print(f"{'='*60}")
    
    has_vertex_colors = (hasattr(mesh, 'vertex_colors') and 
                         mesh.vertex_colors is not None and 
                         len(mesh.vertex_colors) > 0)
    
    has_face_colors = (hasattr(mesh, 'visual') and 
                       hasattr(mesh.visual, 'face_colors') and
                       mesh.visual.face_colors is not None and
                       len(mesh.visual.face_colors) > 0)
    
    has_textures = len(found_textures) > 0
    has_mtl = os.path.exists(mtl_path)
    
    if has_vertex_colors or (has_face_colors and not all_white):
        print("✓ MESH HAS COLORS - Issue may be in Gaussian conversion")
    elif has_textures and has_mtl:
        print("⚠ MESH HAS TEXTURE FILES but not loaded into geometry")
        print("  → Paint pipeline may have generated files but not embedded them")
    elif has_face_colors and all_white:
        print("✗ MESH IS WHITE - Texture was not applied")
        print("  → Check if paint pipeline generated textures")
    else:
        print("✗ MESH HAS NO COLORS")
        print("  → Paint pipeline likely failed or was skipped")
    
    print(f"\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Check if OBJ file has colors/textures applied"
    )
    parser.add_argument("obj_path", help="Path to OBJ file to check")
    args = parser.parse_args()
    
    check_obj_colors(args.obj_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_obj_colors.py <path_to_obj>")
        print("\nExample:")
        print("  python check_obj_colors.py placement_4/test_object.obj")
        sys.exit(1)
    
    main()
