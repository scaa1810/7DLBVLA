import torch
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("🚀 CPU TRAINING (SHORT RUN, 10 epochs)")

data = torch.load("rtx_dummy.pt")
print(f"Training on {len(data)} samples")

rgb_all = torch.stack([d['rgb'] for d in data])
joints_all = torch.zeros(len(data), 14)
actions_all = torch.stack([d['action'] for d in data])
instructions_all = [d['instruction'] for d in data]

device = torch.device("cpu")
print("Using CPU")

from language_biased_vla import LanguageBiasedVLA
model = LanguageBiasedVLA().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)

print("Training 10 epochs...")
batch_size = 16
num_batches = len(data) // batch_size

for epoch in tqdm(range(10)):
    epoch_loss = 0.0
    for i in range(0, len(data), batch_size):
        batch_rgb = rgb_all[i:i+batch_size].to(device)
        batch_joints = joints_all[i:i+batch_size].to(device)
        batch_actions = actions_all[i:i+batch_size].to(device)
        batch_instr = instructions_all[i:i+batch_size]

        pred_actions, text_bias = model(batch_rgb, batch_joints, batch_instr)
        loss = F.mse_loss(pred_actions, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch}: Loss {epoch_loss / max(1, num_batches):.4f}")

torch.save(model.state_dict(), "language_biased_vla_short.pt")
print("✅ TRAINING COMPLETE! Saved: language_biased_vla_short.pt")
