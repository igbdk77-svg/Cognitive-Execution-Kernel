import torch
from boundary_loss import GeometricBoundaryLoss

# 1. Test için yapay zeka parametrelerini simüle edelim (Rastgele Veriler)
batch_size = 4
hidden_dim = 128  # Modelin gizli katman boyutu

# Modelin normal görev kaybı (Örn: Cross Entropy hatası)
loss_task = torch.tensor(1.5, requires_grad=True)

# Modelin o anki katmandaki gizli temsil vektörü (hidden state - h_l)
h_l = torch.randn(batch_size, hidden_dim)

# Yasaklı/Tehlikeli anlamsal alt uzay matrisi (P_perp)
P_perp = torch.randn(hidden_dim, hidden_dim)

# 2. Sizin formülünüzün sınır koruma motorunu tanımlayalım
# Parametreler: tau (eşik)=0.5, sigma=0.1, gamma=0.01, lambda=1.0
safety_engine = GeometricBoundaryLoss(tau=0.5, sigma=0.1, gamma=0.01, lambda_param=1.0)

# 3. Formülü çalıştıralım ve toplam kaybı hesaplayalım
total_loss = safety_engine(loss_task, h_l, P_perp)

print(f"--- Formül ---")
print(f"Normal Görev Kaybı: {loss_task.item()}")
print(f"Formülünüzün Koruma Altındaki Toplam Kaybı: {total_loss.item()}")

# 4. Modeli güvenli bölgeye geri iten gradyan akışını tetikleyelim (Backpropagation)
total_loss.backward()
print("Gradyanlar başarıyla hesaplandı. Model güvenli alt uzaya doğru itiliyor!")
