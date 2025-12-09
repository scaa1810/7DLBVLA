# 7D Language‑Biased VLA for Robot Manipulators

This project tests a simple idea:  
use vision and joint state to predict a robot arm action, then let the **instruction add a 7‑D bias directly on that action** (gripper + 6 arm/wrist axes).

## Core idea

- Baseline:  
  `RGB image + joints → network → 7‑D action`

- 7D Language‑Biased VLA (this project):  
  - `RGB image + joints → base_action (7‑D)`  
  - `instruction text → text_bias (7‑D)`  
  - **final action:** `base_action + text_bias`

So language does not just disappear into deep features: it shows up as an explicit 7‑D offset you can inspect joint‑by‑joint.

## What’s new here

- **Text acts in action space, not only in feature space.**  
  The instruction is turned into a 7‑D vector that directly nudges the gripper and arm/wrist commands.

- **Separation between “what I would do” and “what the instruction wants”:**  
  - `base_action` = vision + state only  
  - `text_bias` = how the instruction corrects or adjusts that action

- **Easy to interpret:**  
  for “open gripper”, the gripper bias goes up; for “move arm left”, the relevant arm axis gets a different bias pattern.

## Why this could be useful

- Robot arms that are easier to debug:  
  you can print the 7 numbers from `text_bias` and see how language is influencing each joint.

- Layer on top of existing policies:  
  the bias head can be seen as a small “instruction layer” on top of a standard vision‑based controller.

- Stepping stone to real RT‑X style data:  
  the same architecture can be dropped into a larger VLA setup to study how much of the behavior can be controlled by simple, low‑dimensional language biases.

## How to run

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
-> for create a small RT‑X‑style dummy dataset (1000 samples): python3 src/make_dummy_data.py
python3 src/train_cpu_short.py
evaluate language‑biased model: python3 src/eval_vla.py
train + evaluate vision‑only baseline: python3 src/baseline_eval.py

## Results (on the small synthetic dataset)

- Vision‑only baseline: **MSE ≈ 0.0868**  
- 7D Language‑Biased VLA: **MSE ≈ 0.0862**

Training is short and the dataset is synthetic, but this is enough to show that:

- adding a 7‑D language bias at the action head is **stable**,  
- it performs at least as well as the baseline, and  
- it produces clear and instruction‑dependent joint‑space biases that can be plotted and inspected.

