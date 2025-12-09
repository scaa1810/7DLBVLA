import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

print("🚀 CPU TRAINING: Language-Biased VLA (RTX 3050 optimized)")
device = torch.device("cpu")

data = torch.load("rtx_dummy.pt")
print(f"Training on {len(data)} samples")

rgb_list = [d['rgb'] for d in data]
action_list = [d['action'] for d in data]
instruction_list = [d['instruction'] for d in data]
joints_list = [torch.zeros(14) for _ in data]

dataset = list(zip(rgb_list, joints_list, instruction_list, action_list))
loader = DataLoader(dataset, batch_size=8, shuffle=True)  # Smaller batch

from language_biased_vla import LanguageBiasedVLA
model = LanguageBiasedVLA().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

print("Training 50 epochs (CPU)...")
for epoch in tqdm(range(50)):
    epoch_loss = 0
    epoch_bias = 0
    
    for batch_rgb, batch_joints, batch_instructions, batch_actions in loader:
        batch_rgb = batch_rgb.to(device)
        batch_joints = torch.stack(batch_joints).to(device)
        batch_actions = torch.stack(batch_actions).to(device)
        
        pred_actions, text_bias = model(batch_rgb, batch_joints, batch_instructions)
        loss = F.mse_loss(pred_actions, batch_actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_bias += text_bias.abs().mean().item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {epoch_loss/len(loader):.4f}, Bias {epoch_bias/len(loader):.3f}")

torch.save(model.state_dict(), "language_biased_vla_cpu.pt")
