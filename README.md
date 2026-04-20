# Robust Robotic Manipulation via Domain Randomization and Soft Actor-Critic

This project investigates sim-to-real transfer for robotic manipulation through 
domain randomization (DR). A pick-and-place policy is trained using Soft Actor-Critic 
(SAC) in MuJoCo with a Franka Panda manipulator, where physics parameters — object 
mass, surface friction, joint damping, and observation noise — are randomized across 
episodes to produce policies robust to environmental uncertainty.

## Key Components
- **Theoretical foundation:** Formulation of domain randomization as optimization 
  over a distribution of MDPs; derivation of the SAC objective (maximum entropy 
  Bellman equation, soft policy improvement); connection to robust MDPs
- **Task:** 7-DoF Franka Panda pick-and-place with delta end-effector control and 
  gated shaped reward
- **Experiments:** Baseline (fixed environment) vs. domain-randomized training, 
  ablation over individual randomization parameters, robustness evaluation under 
  out-of-distribution physics perturbations

## Stack
Python · MuJoCo · Stable-Baselines3 (SAC) · NumPy · Matplotlib

## References
- Haarnoja et al., "Soft Actor-Critic" (ICML 2018)
- Tobin et al., "Domain Randomization for Sim-to-Real Transfer" (2017)
- OpenAI et al., "Solving Rubik's Cube with a Robot Hand" (2019)
- DORAEMON: Entropy-based Domain Randomization (ICLR 2024)
