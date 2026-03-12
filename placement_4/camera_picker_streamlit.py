import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch
import os

# Load checkpoint and scene data
def load_scene(ckpt_path, opacity_threshold=0.1):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["pipeline"]
    means = state["_model.means"].numpy()
    opacities_raw = state["_model.opacities"].numpy()
    opacities = (1 / (1 + np.exp(-opacities_raw))).squeeze()
    vis_mask = opacities > opacity_threshold
    vis_means = means[vis_mask]
    return vis_means

st.title("Interactive 3D Scene Camera Picker")

ckpt_path = st.text_input("Checkpoint path", "bench_park.ckpt")
opacity_threshold = st.slider("Opacity threshold", 0.0, 1.0, 0.1)

if os.path.exists(ckpt_path):
    vis_means = load_scene(ckpt_path, opacity_threshold)
    st.success(f"Loaded {vis_means.shape[0]} visible points.")

    # Plotly 3D scatter
    fig = go.Figure(data=[go.Scatter3d(
        x=vis_means[:,0], y=vis_means[:,1], z=vis_means[:,2],
        mode='markers', marker=dict(size=2, color=vis_means[:,2], colorscale='Viridis')
    )])
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Camera controls
    st.subheader("Camera Controls")
    cam_x = st.slider("Camera X", float(vis_means[:,0].min()), float(vis_means[:,0].max()), float(vis_means[:,0].mean()))
    cam_y = st.slider("Camera Y", float(vis_means[:,1].min()), float(vis_means[:,1].max()), float(vis_means[:,1].mean()))
    cam_z = st.slider("Camera Z", float(vis_means[:,2].min()), float(vis_means[:,2].max()), float(vis_means[:,2].mean()))
    fov = st.slider("Field of View (deg)", 30.0, 120.0, 60.0)

    # Show camera parameters
    st.write("**Selected Camera Position:**", (cam_x, cam_y, cam_z))
    st.write("**Field of View:**", fov)
    # Optionally, add a button to save or use these parameters
    if st.button("Save Camera Parameters"):
        import json
        params = {
            "position": [cam_x, cam_y, cam_z],
            "fov": fov,
            "width": int(fig.layout.scene.xaxis.range[1] - fig.layout.scene.xaxis.range[0]) if fig.layout.scene.xaxis.range else 1280,
            "height": int(fig.layout.scene.yaxis.range[1] - fig.layout.scene.yaxis.range[0]) if fig.layout.scene.yaxis.range else 720
        }
        with open("camera_params.json", "w") as f:
            json.dump(params, f)
        st.success(f"Camera parameters saved to camera_params.json: {params}")
else:
    st.warning("Checkpoint file not found.")
