import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm

class VLADataset(Dataset):
    def __init__(self, data):
        self.rgb = torch.stack([d['rgb'] for d in data])
        self.actions = torch.stack([d['action'] for d in data])
        self.instructions = [d['instruction'] for d in data]
        self.joints = torch.zeros(len(data), 14)
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        return (self.rgb[idx], self.joints[idx], self.instructions[idx], self.actions[idx])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = torch.load("rtx_dummy.pt")
dataset = VLADataset(data)
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)# 8 not 12
from language_biased_vla import LanguageBiasedVLA
model = LanguageBiasedVLA().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

print("training 50 epochs...")
for epoch in tqdm(range(50)):
    model.train()
    epoch_loss = 0
    
    for batch in loader:
        batch_rgb, batch_joints, batch_instructions, batch_actions = batch
        batch_rgb = batch_rgb.to(device, non_blocking=True)
        batch_joints = batch_joints.to(device, non_blocking=True)
        batch_actions = batch_actions.to(device, non_blocking=True)
        
        pred_actions, text_bias = model(batch_rgb, batch_joints, batch_instructions)
        loss = F.mse_loss(pred_actions, batch_actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {epoch_loss/len(loader):.4f}")

torch.save(model.state_dict(), "language_biased_vla_final.pt")
