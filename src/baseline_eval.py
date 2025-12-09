import torch
import torch.nn.functional as F
from baseline_model import BaselineVLA

print("TRAIN+EVAL BaseLine (no lang)")

data = torch.load("rtx_dummy.pt")
rgb_all = torch.stack([d['rgb'] for d in data])
joints_all = torch.zeros(len(data), 14)
actions_all = torch.stack([d['action'] for d in data])

device = torch.device("cpu")
model = BaselineVLA().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

batch_size = 16
for epoch in range(5):
    epoch_loss =0.0
    
    for i in range(0, len(data), batch_size):
        
        b_rgb = rgb_all[i:i+batch_size].to(device)
        b_j = joints_all[i:i+batch_size].to(device)
        b_a = actions_all[i:i+batch_size].to(device)
        pred= model(b_rgb, b_j)
        loss = F.mse_loss(pred, b_a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss+=loss.item()
        
    print(f"Epoch {epoch}: Baseline loss {epoch_loss / max(1, len(data)//batch_size):.4f}")

with torch.no_grad():
    pred= model(rgb_all, joints_all)
    mse = F.mse_loss(pred, actions_all).item()

print(f"Baseline Eval MSE: {mse:.4f}")
