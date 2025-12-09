import torch
import numpy as np
from transformers import CLIPProcessor

def main():
    print("Synthetic RT-X style dummy dataset (creating 1000 samples...)")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    instructions = ["open gripper", "move arm left", "pick up red block"]
    data=[]

    for i in range(1000):

        rgb = np.random.rand(3, 224, 224).astype("float32")
        action = np.random.rand(7).astype("float32")
        instr = np.random.choice(instructions)
        inputs = proc(text=[instr], return_tensors="pt", padding=True)
        data.append({"rgb": torch.tensor(rgb),"action": torch.tensor(action),"instruction": instr,"instruction_ids": inputs["input_ids"][0],"instruction_mask": inputs["attention_mask"][0],})
    torch.save(data, "data/rtx_dummy.pt")
    print("\n saved at data/rtx_dummy.pt")

if __name__ == "__main__":
    main()
