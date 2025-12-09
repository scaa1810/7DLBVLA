import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

data = torch.load("rtx_dummy.pt")
print(f"Training on {len(data)} samples")
rgb_list = [d['rgb'] for d in data]
action_list = [d['action'] for d in data]
instruction_list = [d['instruction'] for d in data]
joints_list = [torch.zeros(14) for _ in data]  #dummy joints

dataset = list(zip(rgb_list, joints_list, instruction_list, action_list))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

from language_biased_vla import LanguageBiasedVLA
model = LanguageBiasedVLA().cuda()
optimizer = Adam(model.parameters(), lr=1e-4)

losses = []
biases = []

print("Training 100 epochs...")
for epoch in tqdm(range(100)):
    epoch_loss =epoch_bias_mag = 0

    for batch_rgb, batch_joints, batch_instructions, batch_actions in loader:
        
        batch_rgb = batch_rgb.cuda()
        batch_joints = batch_joints.cuda()
        batch_actions = batch_actions.cuda()
        pred_actions, text_bias = model(batch_rgb, batch_joints, batch_instructions)
        loss = F.mse_loss(pred_actions, batch_actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_bias_mag += text_bias.abs().mean().item()
    losses.append(epoch_loss / len(loader))
    biases.append(epoch_bias_mag / len(loader))
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss {losses[-1]:.4f}, Bias mag {biases[-1]:.3f}")

torch.save(model.state_dict(), "language_biased_vla_trained.pt")
torch.save({'losses': losses, 'biases': biases}, "training_history.pt")
print("Model saved: language_biased_vla_trained.pt")
