# GRPO vs PPO for LLM-based Game Agents in TextWorld

**Course:** CIS 4930 — Deep Reinforcement Learning | Spring 2026 | University of South Florida  
**Authors:** Tasnim Tabassum, Xiangfei Kong, Chibuike Ezeokoli  

## Overview
This repository contains the code and report for our project investigating whether Group Relative Policy Optimization (GRPO) can outperform Proximal Policy Optimization (PPO) when fine-tuning large language models (LLMs) as game-playing agents in text-based environments. We utilized TextWorld (CoinCollector) and the Qwen2.5-1.5B-Instruct model for our experiments.

## Methodology
Our training pipeline implements a shared Supervised Fine-Tuning (SFT) warm start followed by:
1. Behavioral Cloning (BC) baseline
2. PPO Fine-Tuning
3. Multiple GRPO Fine-Tuning variants (testing group sizes G=8 and G=64)

## Key Results
* **PPO & BC Baselines:** PPO achieved a 22% completion rate, while BC achieved 12%.
* **GRPO (G=8):** Our initial GRPO v4 setting reached 10% completion, falling below both baselines and demonstrating GRPO's high sensitivity to reward design, data structure, and action-interface errors.
* **GRPO (G=64):** An extended GRPO run utilizing a larger group size and evaluation-time controls (output cleaning, recent-action history, no-repeat logic, and fallback action selection) achieved **23% completion** under greedy decoding, slightly outperforming PPO.
* **Ablations:** Further evaluation-time ablations revealed that stochastic decoding combined with larger step budgets can push GRPO G=64 completion rates as high as 47% on dense games and 50% on sparse games.

## Repository Contents
* `GRPO_TextWorld_Complete (2).ipynb`: Primary project notebook containing the pipeline.
* `GRPO_retrain (1).ipynb`: Notebook for retraining runs.
* `GRPO_Report_Updated_Keep_G8_G64_Ablation_FullTable (2).pdf`: Full project report detailing the methodology, development iterations, and complete ablation tables.
