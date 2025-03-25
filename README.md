# Deep Reinforcement Learning with PyTorch – Policy Gradient

This repository implements core **Policy Gradient** algorithms in deep reinforcement learning using **PyTorch**. The goal is to train agents to solve environments like `MountainCar-v0` using **REINFORCE** and its variants.

---

## Overview

Policy Gradient methods directly optimize the agent’s policy by following the **gradient of expected rewards**. This allows learning in environments with **high-dimensional** or **continuous action spaces**.

This project demonstrates:

- The basic **REINFORCE** algorithm  
- **REINFORCE with a baseline** to reduce variance  
- A **naive implementation** for learning purposes  
- Structured **training and model execution logic**

---

## Environment

- **Environment**: `MountainCar-v0` from [OpenAI Gym](https://gym.openai.com/)
- **Objective**: The agent must learn to **swing the car** to reach the top of the hill
- **Reward Structure**: Sparse and negative until the goal is reached

---

## Learning Goals

- Understand how **Policy Gradient** algorithms work
- Learn how **REINFORCE** performs updates using full-episode returns
- Explore how adding a **baseline** improves learning efficiency
- Gain hands-on experience using **PyTorch** in reinforcement learning

---

## File Descriptions

### `PolicyGradient.py`
Encapsulates the `PolicyGradient` class:
- Neural network-based policy
- Action sampling from probability distribution
- Reward collection and policy updates via gradient ascent

### `REINFORCE.py`
Implements the **vanilla REINFORCE** algorithm:
- Collects full episodes
- Computes returns
- Updates the policy using log-likelihood gradient

### `REINFORCE_with_Baseline.py`
Extends REINFORCE with a **value function baseline**:
- Reduces gradient variance
- Speeds up convergence and learning stability

### `Run_Model.py`
Runs the complete **training pipeline**:
- Initializes environment and agent
- Executes training loop
- Logs performance metrics

### `naive-policy-gradient.py`
A **minimal policy gradient** implementation:
- Focused on educational clarity
- Helps in understanding core algorithm without abstractions

### `pytorch_MountainCar-v0.py`
Sets up the **MountainCar-v0** environment:
- Defines state and action space
- Used to test trained policy gradient agents

---

## Requirements

- Python 3.x
- `torch`
- `gym`
- `numpy`
- (Optional) `matplotlib` for visualizations

Install with:

\\\ bash
pip install torch gym numpy matplotlib
\\\

---

## 🏁 Getting Started

To train the REINFORCE agent:

\\\bash
python Run_Model.py
\\\

To run the naive version:

\\\bash
python naive-policy-gradient.py
\\\

---

## References

- [Policy Gradient Methods for RL](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- OpenAI Gym documentation: https://www.gymlibrary.dev

---

## 📬 Contact

Feel free to open issues or pull requests if you find any bugs or want to contribute!
