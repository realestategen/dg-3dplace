import torch
import math

def get_projection_matrix(znear, zfar, fovX, fovY):
    """Generates standard OpenGL projection matrix."""
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class CameraWrapper:
    """A struct to hold the camera parameters for the rasterizer."""
    def __init__(self, data_dict, device="cuda"):
        self.device = device
        self.image_width = int(data_dict["render_width"])
        self.image_height = int(data_dict["render_height"])
        self.FoVy = float(data_dict["fov_rad"])
        
        # Extract position (Convert from numpy to tensor and FORCE 32-bit float)
        self.camera_center = torch.as_tensor(data_dict["position"]).float().to(device)
        
        # Extract Extrinsics
        w2c = torch.as_tensor(data_dict["extrinsics_w2c"]).float()
        if w2c.shape == (3, 4):
            bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=w2c.device)
            w2c = torch.cat([w2c, bottom_row], dim=0)
            
        self.world_view_transform = w2c.transpose(0, 1).to(device)
        
        # Calculate projection for the first time
        self._update_projection()

    def _update_projection(self):
        """Recalculates the lens math based on the current width and height."""
        aspect_ratio = self.image_width / self.image_height
        self.FoVx = 2 * math.atan(math.tan(self.FoVy / 2) * aspect_ratio)
        proj = get_projection_matrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy).to(self.device)
        self.full_proj_transform = torch.matmul(self.world_view_transform, proj)

    def update_resolution(self, new_width, new_height):
        """Safely updates the resolution and recalculates the lens math."""
        self.image_width = new_width
        self.image_height = new_height
        self._update_projection()

def load_scout_camera(filepath, device="cuda"):
    """Loads the camera data saved by the Camera Scout."""
    try:
        data_dict = torch.load(filepath, map_location=device)
        return CameraWrapper(data_dict, device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find camera file at {filepath}")