import torch
import torch.nn as nn

class GeometricBoundaryLoss(nn.Module):
    def __init__(self, tau, sigma, gamma, lambda_param):
        super(GeometricBoundaryLoss, self).__init__()
        self.tau = tau
        self.sigma = sigma
        self.gamma = gamma
        self.lambda_param = lambda_param

    def forward(self, loss_task, h_l, P_perp):
        # Hidden state (h_l) vektörünün yasaklı P_perp alt uzayına dik izdüşümü
        proj = torch.matmul(h_l, P_perp)
        norm_proj = torch.norm(proj, p=2, dim=-1)
        
        # Sigmoid Kapısı (Yazılımsal Tetikleyici)
        gate = torch.sigmoid((norm_proj - self.tau) / self.sigma)
        
        # Üstel Patlama Cezası
        penalty = torch.exp(self.gamma * (norm_proj ** 2))
        
        # Toplam Kayıp Fonksiyonu (Formülünüzün Karşılığı)
        loss_total = loss_task + (self.lambda_param * gate * penalty)
        return loss_total.mean()
