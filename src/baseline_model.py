import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection

class BaselineVLA(nn.Module):
    """Vision + joints → action (no language)"""
    def __init__(self):
        super().__init__()
        self.vision = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        dim = self.vision.config.projection_dim
        self.head = nn.Sequential(
            nn.Linear(dim + 14, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, rgb, joints):
        v_feat = self.vision(pixel_values=rgb).image_embeds
        fused = torch.cat([v_feat, joints], dim=1)
        return self.head(fused)
