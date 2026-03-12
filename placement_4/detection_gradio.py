import gradio as gr
import numpy as np
import math
import os
from detection_optimized import (
    load_scene,  # You should have a function to load your scene
    render_camera_view,  # You should have a function to render a camera view and return a PIL image
    run_rest_of_pipeline  # You should have a function to run the rest of your script after camera selection
)

# Default presets from detection_optimized.py
FOV_DEG = 60.0
CAMERA_HEIGHT_OFFSET = 0.0
ORBIT_SCALE = 0.0005

# You may need to adjust these imports and function names to match your codebase

scene = load_scene()  # Load your scene once


def get_camera_image(fov_deg, camera_height_offset, azimuth_angle_deg):
    # Convert azimuth to radians
    azimuth_angle = math.radians(azimuth_angle_deg)
    # Compute orbit radius as in your script
    scene_extent = np.array([1.0, 1.0, 1.0])  # Replace with actual scene extent
    orbit_radius = float(np.linalg.norm(scene_extent)) * ORBIT_SCALE
    # Render the camera view
    img = render_camera_view(
        scene=scene,
        orbit_radius=orbit_radius,
        camera_height_offset=camera_height_offset,
        azimuth_angle=azimuth_angle,
        fov_deg=fov_deg
    )
    return img


def submit_and_save(fov_deg, camera_height_offset, azimuth_angle_deg):
    img = get_camera_image(fov_deg, camera_height_offset, azimuth_angle_deg)
    save_path = f"selected_camera_{fov_deg:.1f}_{camera_height_offset:.2f}_{azimuth_angle_deg:.1f}.png"
    img.save(save_path)
    # Run the rest of your pipeline
    run_rest_of_pipeline(
        scene=scene,
        fov_deg=fov_deg,
        camera_height_offset=camera_height_offset,
        azimuth_angle_deg=azimuth_angle_deg,
        save_path=save_path
    )
    return f"Saved and processed: {save_path}"


def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Interactive Camera Selection for 3D Scene")
        with gr.Row():
            fov_slider = gr.Slider(10, 120, value=FOV_DEG, label="FOV (degrees)")
            height_slider = gr.Slider(-1.0, 2.0, value=CAMERA_HEIGHT_OFFSET, label="Camera Height Offset")
            azimuth_slider = gr.Slider(0, 360, value=0, label="Azimuth Angle (degrees)")
        img_output = gr.Image(label="Camera View")
        preview_btn = gr.Button("Preview Camera View")
        submit_btn = gr.Button("Submit and Save Selected View")
        status = gr.Textbox(label="Status")

        preview_btn.click(
            fn=get_camera_image,
            inputs=[fov_slider, height_slider, azimuth_slider],
            outputs=img_output
        )
        submit_btn.click(
            fn=submit_and_save,
            inputs=[fov_slider, height_slider, azimuth_slider],
            outputs=status
        )
    return demo


def main():
    demo = gradio_interface()
    demo.launch(server_port=7860, share=True)  # Open on port 7860

if __name__ == "__main__":
    main()
