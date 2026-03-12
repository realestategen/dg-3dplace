import streamlit as st
import numpy as np
import math
from PIL import Image
import torch
import os
import json
from detection_optimized import render_single_camera

# Load scene info for camera placement
CKPT_PATH = "bench_park.ckpt"
RENDER_W, RENDER_H = 1280, 720
ORBIT_SCALE = 0.0005
CAMERA_HEIGHT_OFFSET = 0.1
FOV_DEG = 60.0
SESSION_DIR = "session_streamlit"
os.makedirs(SESSION_DIR, exist_ok=True)

# Dummy scene_center and scene_extent for demo (replace with real values)
scene_center = np.array([0.0, 0.0, 0.0])
scene_extent = np.array([1.0, 1.0, 1.0])

st.set_page_config(layout="wide")
st.title("Interactive 3D Camera Navigation (Streamlit Demo)")

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### Camera Controls")
    fov_deg = st.slider("FOV (degrees)", 10, 120, int(FOV_DEG))
    camera_height_offset = st.slider("Camera Height Offset", -1.0, 2.0, float(CAMERA_HEIGHT_OFFSET))
    azimuth_deg = st.slider("Azimuth Angle (degrees)", 0, 360, 0)
    elevation_deg = st.slider("Elevation (degrees)", -90, 90, 0)
    orbit_radius_scale = st.slider("Orbit Radius Scale", 0.2, 3.0, 1.0)
    lookat_x = st.number_input("Look-at X", value=float(scene_center[0]))
    lookat_y = st.number_input("Look-at Y", value=float(scene_center[1]))
    lookat_z = st.number_input("Look-at Z", value=float(scene_center[2]))
    if st.button("Save Camera View"):
        img = render_single_camera(scene_center, scene_extent, azimuth_deg, elevation_deg, orbit_radius_scale, fov_deg, camera_height_offset, lookat_x, lookat_y, lookat_z)
        save_base = f"selected_camera_{fov_deg:.1f}_{camera_height_offset:.2f}_{azimuth_deg:.1f}_{elevation_deg:.1f}_{orbit_radius_scale:.2f}"
        save_path = os.path.join(SESSION_DIR, save_base + ".png")
        img.save(save_path)
        # Save camera parameters as JSON
        cam_params = {
            "fov_deg": fov_deg,
            "camera_height_offset": camera_height_offset,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "orbit_radius_scale": orbit_radius_scale,
            "lookat_x": lookat_x,
            "lookat_y": lookat_y,
            "lookat_z": lookat_z
        }
        with open(os.path.join(SESSION_DIR, save_base + ".json"), "w") as f:
            json.dump(cam_params, f, indent=2)
        st.success(f"Saved: {save_path} and camera parameters JSON.")

with col1:
    st.markdown("### Camera Preview")
    img = render_single_camera(scene_center, scene_extent, azimuth_deg, elevation_deg, orbit_radius_scale, fov_deg, camera_height_offset, lookat_x, lookat_y, lookat_z)
    st.image(img, caption="Camera View", use_column_width=True)
    st.caption("Drag the sliders or enter values to move the camera. For true 3D drag, see a three.js or vtk.js web viewer.")
