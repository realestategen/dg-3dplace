#!/usr/bin/env python3
"""Quick test to see what's in the GLB file."""
import os
import trimesh
import numpy as np
from PIL import Image as PILImage

# Check what we generated recently
session_dir = "."
for f in sorted(os.listdir(session_dir), reverse=True):
    if f.endswith('.glb') or f.endswith('.obj'):
        mesh_path = os.path.join(session_dir, f)
        print(f"\n{'='*60}")
        print(f"Testing: {f}")
        print(f"{'='*60}")
        
        if not os.path.getsize(mesh_path) > 0:
            print(f"File is empty, skipping")
            continue
        
        try:
            mesh = trimesh.load(mesh_path, process=False)
            print(f"✓ Loaded with trimesh")
            print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
            
            # Check visual attributes
            if hasattr(mesh, 'visual'):
                visual = mesh.visual
                print(f"  Visual type: {type(visual).__name__}")
                
                if hasattr(visual, 'vertex_colors'):
                    vc = getattr(visual, 'vertex_colors', None)
                    if vc is not None:
                        print(f"  ✓ Has vertex_colors: {vc.shape}")
                    else:
                        print(f"  ✗ vertex_colors is None")
                
                if hasattr(visual, 'face_colors'):
                    fc = getattr(visual, 'face_colors', None)
                    if fc is not None:
                        print(f"  ✓ Has face_colors: {fc.shape}")
                    else:
                        print(f"  ✗ face_colors is None")
                
                if hasattr(visual, 'uv'):
                    uv = getattr(visual, 'uv', None)
                    if uv is not None:
                        print(f"  ✓ Has UV coordinates: {uv.shape}")
                    else:
                        print(f"  ✗ UV is None")
                
                if hasattr(visual, 'material'):
                    mat = getattr(visual, 'material', None)
                    if mat is not None:
                        print(f"  ✓ Has material: {type(mat).__name__}")
                        if hasattr(mat, 'image'):
                            img = getattr(mat, 'image', None)
                            if img is not None:
                                print(f"    ✓ Material has image: {np.asarray(img).shape}")
                            else:
                                print(f"    ✗ Material.image is None")
                    else:
                        print(f"  ✗ material is None")
            else:
                print(f"  ✗ No visual attribute")
            
            # Try to extract a sample color
            if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
                sample_face_idx = 0
                face = mesh.faces[sample_face_idx]
                print(f"\n  Sample face {sample_face_idx}: {face}")
                
                # Try to bake by sampling
                if hasattr(mesh, 'export'):
                    try:
                        test_export = mesh_path.replace('.glb', '_test.obj').replace('.obj', '_test.obj')
                        mesh.export(test_export, include_normal=True)
                        reloaded = trimesh.load(test_export, process=False)
                        print(f"  ✓ Export/reload succeeded, reloaded visual type: {type(reloaded.visual).__name__}")
                        if hasattr(reloaded.visual, 'vertex_colors'):
                            vc = getattr(reloaded.visual, 'vertex_colors', None)
                            if vc is not None:
                                print(f"    ✓ Reloaded has vertex_colors: {vc.shape}")
                                print(f"    Sample color: {vc[face[0]]}")
                            else:
                                print(f"    ✗ Reloaded vertex_colors is None")
                    except Exception as e:
                        print(f"  ✗ Export failed: {e}")
        
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            import traceback
            traceback.print_exc()
        
        # Only test first mesh file
        break
