import panel as pn
import pyvista as pv
import numpy as np
import torch
import json
import os

pn.extension('vtk')

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

ckpt_path = "bench_park.ckpt"
opacity_threshold = 0.1

if os.path.exists(ckpt_path):
    vis_means = load_scene(ckpt_path, opacity_threshold)
    cloud = pv.PolyData(vis_means)
    plotter = pv.Plotter()
    plotter.add_points(cloud, color="white", point_size=2)
    plotter.background_color = "black"
    plot_widget = plotter.show(return_viewer=True, interactive=True, screenshot=False)

    def save_camera_params(event=None):
        cam = plotter.camera
        params = {
            "position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "view_up": list(cam.up),
            "fov": cam.view_angle,
            "width": 1280,
            "height": 720
        }
        with open("camera_params.json", "w") as f:
            json.dump(params, f)
        status.value = f"Camera parameters saved: {params}"

    save_button = pn.widgets.Button(name="Save Camera Parameters", button_type="primary")
    save_button.on_click(save_camera_params)
    status = pn.pane.Markdown("")

    app = pn.Column(plot_widget, save_button, status)
    app.servable()
else:
    pn.pane.Markdown("Checkpoint not found.").servable()
