import torch
import torch.nn as nn

def quaternion_to_rotation_matrix(r):
    """Converts a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    norm = torch.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3])
    q = r / norm
    R = torch.zeros((3, 3), device=q.device)
    R[0, 0] = 1 - 2 * (q[2]**2 + q[3]**2)
    R[0, 1] = 2 * (q[1]*q[2] - q[0]*q[3])
    R[0, 2] = 2 * (q[1]*q[3] + q[0]*q[2])
    R[1, 0] = 2 * (q[1]*q[2] + q[0]*q[3])
    R[1, 1] = 1 - 2 * (q[1]**2 + q[3]**2)
    R[1, 2] = 2 * (q[2]*q[3] - q[0]*q[1])
    R[2, 0] = 2 * (q[1]*q[3] - q[0]*q[2])
    R[2, 1] = 2 * (q[2]*q[3] + q[0]*q[1])
    R[2, 2] = 1 - 2 * (q[1]**2 + q[2]**2)
    return R

class PoseOptimizer(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # Pose parameters to learn
        self.translation = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
        # Initialize quaternion as identity [w, x, y, z]
        self.rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device))
        self.scale = nn.Parameter(torch.ones(3, dtype=torch.float32, device=device))
        
    def transform_object(self, obj_gaussians):
        """
        Applies the current translation, rotation, and scale to the object Gaussians.
        """
        R = quaternion_to_rotation_matrix(self.rotation)
        
        transformed_obj = {}
        
        # 1. Update Positions (Means)
        transformed_obj['means'] = torch.matmul(obj_gaussians['means'], (R * self.scale).T) + self.translation
        
        # 2. Update Scales
        if 'scales' in obj_gaussians:
            # Note: 3DGS scales are usually stored as log(scale)
            # You may need to adapt this depending on how your specific 3DGS implementation stores them
            transformed_obj['scales'] = obj_gaussians['scales'] + torch.log(self.scale)
            
        # 3. Pass through remaining properties unchanged for this basic pose optimization
        for key in ['colors', 'opacities', 'rotations', 'features_dc', 'features_rest']:
            if key in obj_gaussians:
                transformed_obj[key] = obj_gaussians[key]
                
        return transformed_obj
        
    def normalize_quaternion(self):
        """Must be called after optimizer.step() to keep the rotation valid."""
        with torch.no_grad():
            self.rotation.div_(torch.norm(self.rotation))