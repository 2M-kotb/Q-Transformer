# Q-Transformer
<img src="https://github.com/2M-kotb/Q-Transformer/blob/main/QT.png" width=50% height=50%>

This is an adaptive version of [Q-Transformer](https://qtransformer.github.io/) model from Google Deepmind, in which it only works with state-based tasks in an online RL scenario.
This model used in the following paper:

[QT-TDM:Planning with Transformer Dynamics Model and Autoregressive Q-Learning](https://arxiv.org/pdf/2407.18841)


# Instructions
Install dependencies using ``` conda ```:
```
conda env create -f environment.yaml
conda activate qt
```
Train the model by calling:
```
python3 main.py env.domain=metaworld env.task=mw-hammer env.action_repeat=2 env.seed=1
```
``` env.domain``` can take ```metaworld``` or ```dmc_suite``` for [MetaWorld](https://meta-world.github.io) and [DeepMind Control Suite](https://github.com/deepmind/dm_control).

To use [Weights&Biases](https://wandb.ai/site/) for logging, set up ```wandb``` variables inside ```config.yaml```.

