# Q-Transformer
<img src="https://github.com/2M-kotb/Q-Transformer/blob/main/QT.png" width=40% height=40%>

This is an adaptive version of [Q-Transformer](https://qtransformer.github.io/) model from Google Deepmind, in which it only works with state-based tasks in an online RL scenario.
This model is used in the following paper:

[QT-TDM: Planning with Transformer Dynamics Model and Autoregressive Q-Learning](https://arxiv.org/pdf/2407.18841v2)

__Paper GitHub:__ [QT-TDM](https://github.com/2M-kotb/QT-TDM/tree/main)


# Instructions
Install dependencies using ``` conda ```:
```
conda env create -f environment.yaml
conda activate qt
```
Train the model by calling:
```
python3 src/main.py env.domain=metaworld env.task=mw-hammer env.action_repeat=2 env.seed=1
```
``` env.domain``` can take ```metaworld``` or ```dmc_suite``` for [MetaWorld](https://meta-world.github.io) and [DeepMind Control Suite](https://github.com/deepmind/dm_control).

For ```dmc_suite``` set ```update_freq: 5``` in ```config.yaml```.

For ```sparse reward tasks``` set ```use_MC_return: true``` in ```config.yaml```.

To use [Weights&Biases](https://wandb.ai/site/) for logging, set up ```wandb``` variables inside ```config.yaml```.

# Citation
cite the paper as follows:
```
@article{kotb2024qt,
  title={QT-TDM: Planning with Transformer Dynamics Model and Autoregressive Q-Learning},
  author={Kotb, Mostafa and Weber, Cornelius and Hafez, Muhammad Burhan and Wermter, Stefan},
  journal={arXiv preprint arXiv:2407.18841},
  year={2024}
}
```

# Credits
* IRIS: https://github.com/eloialonso/iris/tree/main
* minGPT: https://github.com/karpathy/minGPT
* Unofficial implementation of Q-Transformer: https://github.com/lucidrains/q-transformer

