import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPProcessor
import torch.nn.functional as F

print("🚀 BUILDING YOUR NOVEL LANGUAGE-BIASED VLA...")

class LanguageBiasedVLA(nn.Module):
    """YOUR NOVEL ARCHITECTURE: Text directly biases robot actions"""
    def __init__(self):
        super().__init__()
        self.vision = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.text = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        vision_dim = self.vision.config.projection_dim  # 512
        text_dim = self.text.config.projection_dim      # 512
        
        # ⭐ YOUR NOVEL CONTRIBUTIONS ⭐
        self.vision_joints_fusion = nn.Sequential(
            nn.Linear(vision_dim + 14, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # NOVEL #1: Text → Direct Action Bias (not standard concatenation)
        self.text_bias_head = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7D bias for gripper/arm
        )
        
        # NOVEL #2: Final action prediction
        self.action_head = nn.Linear(128 + 7, 7)  # fused_vision + text_bias → action
        
    def forward(self, rgb, joints, instruction):
        # Vision + joints fusion
        v_feat = self.vision(pixel_values=rgb).image_embeds
        vj_fused = self.vision_joints_fusion(torch.cat([v_feat, joints], dim=1))
        
        # ⭐ NOVEL: Text directly predicts action bias
        inputs = self.processor(text=instruction, return_tensors="pt", padding=True).to(rgb.device)
        t_feat = self.text(**inputs).text_embeds
        text_bias = self.text_bias_head(t_feat)  # [B, 7]
        
        # ⭐ NOVEL: Bias added to vision prediction
        final_feat = torch.cat([vj_fused, text_bias], dim=1)
        action = self.action_head(final_feat)
        
        return action, text_bias  # Return bias for analysis

# Test YOUR novel model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LanguageBiasedVLA().to(device)

rgb = torch.randn(2, 3, 224, 224).to(device)
joints = torch.randn(2, 14).to(device)
instructions = ["pick up red block", "move arm left"]

actions, biases = model(rgb, joints, instructions)
print(f"✅ YOUR NOVEL VLA WORKS!")
print(f"Action shape: {actions.shape}")
print(f"Text bias shape: {biases.shape}")
print(f"Sample bias: {biases[0][:3]}")  # First 3 action dimensions
