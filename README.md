# Deep Reinforcement Learning with PyTorch – Policy Gradient
This repository implements core policy gradient algorithms in deep reinforcement learning using PyTorch. The goal is to train agents to solve environments like MountainCar-v0 using REINFORCE and its variants.
## Overview
Policy Gradient methods directly optimize the agent’s policy by following the gradient of expected rewards. This allows learning in environments with high-dimensional or continuous action spaces.
This project demonstrates:
• The basic REINFORCE algorithm
• REINFORCE with a baseline to reduce variance
• A naive implementation for learning purposes
• Structured training and model-running logic

## Environment
• MountainCar-v0 from OpenAI Gym
• Agent must learn to swing the car to reach the top of the hill
• Rewards are sparse and negative until the goal is reached

## Learning Goals
• Understand how policy gradients work
• Learn how REINFORCE performs updates based on full-episode returns
• See how adding a baseline improves learning
• Get hands-on experience with PyTorch in RL



## File Descriptions
PolicyGradient.py
Defines the PolicyGradient class that encapsulates:
• A neural network-based policy
• Action sampling from the probability distribution
• Reward collection and policy updates using gradient ascent

## REINFORCE.py
Implements the vanilla REINFORCE algorithm:
• Collects full episodes
• Computes returns
• Updates policy network using log-likelihood gradient

## REINFORCE_with_Baseline.py
Enhances REINFORCE with a value function baseline:
• Reduces variance in gradients
• Speeds up learning stability and convergence

## Run_Model.py
Runs the training pipeline:
• Initializes environment and agent
• Executes training loop
• Logs performance metrics

## naive-policy-gradient.py
A simplified policy gradient implementation:
• Minimal version for educational use
• Useful for understanding the core algorithm without abstraction
pytorch_MountainCar-v0.py
Sets up the OpenAI Gym MountainCar-v0 environment:
• Defines state and action space
• Used to test agents trained with policy gradient methods
