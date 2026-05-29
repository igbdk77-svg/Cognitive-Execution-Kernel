import torch

class BilissselCekirdek:
    def __init__(self, tau=0.1, sigma=0.05, gamma=1.0):
        self.tau = tau
        self.sigma = sigma
        self.gamma = gamma

    def ceza_hesapla(self, h_l, P_perp):
        """
        Modelin latent (gizli) katmanlarındaki sapmayı hesaplar.
        h_l: Modelin o anki durumu (activation)
        P_perp: Güvenli alan dışına projeksiyon
        """
        # Ortogonal projeksiyonu al
        h_perp = torch.matmul(h_l, P_perp)
        norm = torch.norm(h_perp)
        
        #  formül: Sigmoid tetikleyici * Üstel ceza
        sigmoid_kismi = torch.sigmoid((norm - self.tau) / self.sigma)
        ceza = torch.exp(self.gamma * (norm**2))
        
        return sigmoid_kismi * ceza
        
