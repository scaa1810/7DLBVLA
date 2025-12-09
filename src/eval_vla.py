import torch
import torch.nn.functional as F
from language_biased_vla import LanguageBiasedVLA

print("Eval of the lang-biased VLA...")
data = torch.load("rtx_dummy.pt")
rgb_all = torch.stack([d['rgb'] for d in data])
joints_all = torch.zeros(len(data), 14)
actions_all = torch.stack([d['action'] for d in data])
instructions_all = [d['instruction'] for d in data]
device = torch.device("cpu")
model = LanguageBiasedVLA().to(device)
model.load_state_dict(torch.load("language_biased_vla_short.pt", map_location=device))
model.eval()
with torch.no_grad():
    
    pred_actions, text_bias = model(rgb_all, joints_all, instructions_all)
    mse = F.mse_loss(pred_actions, actions_all).item()

print(f"Eval MSE on 1000 samples {mse:.4f}")
print("\n Example instruction | bias | action:")
for i in range(3):
    print(f"- Instr: {instructions_all[i]!r}")
    print(f"  Bias[0:3]: {text_bias[i][:3]}")
    print(f"  Pred[0:3]: {pred_actions[i][:3]}")
    print(f"  GT[0:3]:   {actions_all[i][:3]}")
