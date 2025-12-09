import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

data = torch.load("rtx_dummy.pt")
print(f"Training on {len(data)} samples")

rgb_all = torch.stack([d['rgb'] for d in data])
joints_all = torch.zeros(len(data), 14)
actions_all = torch.stack([d['action'] for d in data])
instructions_all = [d['instruction'] for d in data]

device = torch.device("cpu")
print("Using CPU (RTX 3050 safe)")

from language_biased_vla import LanguageBiasedVLA
model = LanguageBiasedVLA().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

print("Training 30 epochs...")
for epoch in tqdm(range(30)):
    epoch_loss = 0
    #manual mini-batches w/ b_s =16
    for i in range(0, len(data), 16):
        batch_rgb = rgb_all[i:i+16].to(device)
        batch_joints = joints_all[i:i+16].to(device)
        batch_instructions = instructions_all[i:i+16]
        batch_actions = actions_all[i:i+16].to(device)
        
        pred_actions, text_bias = model(batch_rgb, batch_joints, batch_instructions)
        loss = F.mse_loss(pred_actions, batch_actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss {epoch_loss/(len(data)//16):.4f}")

torch.save(model.state_dict(), "language_biased_vla_final.pt")
print("Model training completed as weel as saved @ language_biased_vla_final.pt")
