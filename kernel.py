import torch

def apply_kernel(h_l, task_loss, lambda_val=0.01, tau=1.0, sigma=0.1, gamma=0.05):
    # Proj_P_perp hesaplama simülasyonu
    proj_norm = torch.norm(h_l) 
    # Sigmoid aktivasyon
    activation = torch.sigmoid((proj_norm - tau) / sigma)
    # Dinamik direnç
    resistance = torch.exp(gamma * (proj_norm**2))
    return task_loss + (lambda_val * activation * resistance)
  
