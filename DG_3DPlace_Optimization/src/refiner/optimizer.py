import torch
import torch.nn as nn

# ==========================================
# [1] FIXED POSE OPTIMIZER (Uniform Scale Only)
# ==========================================
def quaternion_to_rotation_matrix(r):
    norm = torch.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3] + 1e-8)
    q = r / norm
    
    R00 = 1 - 2 * (q[2]**2 + q[3]**2)
    R01 = 2 * (q[1]*q[2] - q[0]*q[3])
    R02 = 2 * (q[1]*q[3] + q[0]*q[2])
    
    R10 = 2 * (q[1]*q[2] + q[0]*q[3])
    R11 = 1 - 2 * (q[1]**2 + q[3]**2)
    R12 = 2 * (q[2]*q[3] - q[0]*q[1])
    
    R20 = 2 * (q[1]*q[3] - q[0]*q[2])
    R21 = 2 * (q[2]*q[3] + q[0]*q[1])
    R22 = 1 - 2 * (q[1]**2 + q[2]**2)
    
    return torch.stack([R00, R01, R02, R10, R11, R12, R20, R21, R22]).reshape(3, 3)

def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)

class PoseOptimizer(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # We will initialize this to the object's TRUE center before the loop
        self.translation = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=device))
        self.rotation = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device))
        self.scale_scalar = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        
        # We will set this manually from the main script
        self.obj_center = None 
        
    def transform_object(self, obj_gaussians):
        R = quaternion_to_rotation_matrix(self.rotation)
        transformed_obj = {}
        
        # 1. Center the object locally using its true starting center
        centered_means = obj_gaussians['means'] - self.obj_center
        
        # 2. Scale and Rotate
        s = torch.abs(self.scale_scalar) + 1e-8
        scaled_rotated_means = torch.matmul(centered_means, (R * s).T)
        
        # 3. Translate it back to the world. 
        # (Since self.translation starts identically to self.obj_center, it won't move on Epoch 0!)
        transformed_obj['means'] = scaled_rotated_means + self.translation
        
        if 'scales' in obj_gaussians:
            transformed_obj['scales'] = obj_gaussians['scales'] + torch.log(s)
            
        if 'rotations' in obj_gaussians:
            normed_rot = self.rotation / torch.norm(self.rotation)
            global_rot_expanded = normed_rot.unsqueeze(0).expand(obj_gaussians['rotations'].shape[0], -1)
            transformed_obj['rotations'] = multiply_quaternions(global_rot_expanded, obj_gaussians['rotations'])
            
        for key in ['colors', 'opacities', 'features_dc', 'features_rest']:
            if key in obj_gaussians:
                transformed_obj[key] = obj_gaussians[key]
                
        return transformed_obj
        
    def normalize_quaternion(self):
        with torch.no_grad():
            self.rotation.div_(torch.norm(self.rotation) + 1e-8)
