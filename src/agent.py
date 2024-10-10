from copy import deepcopy
from random import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from einops import pack, unpack, repeat, reduce, rearrange
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from utils import ema, LinearSchedule, batch_select_indices
from QTransformer import QTransformer




class Agent(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self.device = torch.device(cfg.misc.device)
        # Q-Transformer
        self.q_transformer = QTransformer(cfg.qtransformer).to(self.device)
        # EMA model (Q-Traget)
        self.ema_model = deepcopy(self.q_transformer).requires_grad_(False)
        # optimizer
        self.optimizer = torch.optim.Adam(self.q_transformer.parameters(), lr=cfg.qtransformer.lr, eps=cfg.qtransformer.eps, weight_decay=cfg.qtransformer.decay) 
        # learning rate decay used in metaworld 
        if cfg.env.domain == "metaworld":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: (1 - (steps/self.cfg.env.train_steps)))   

        # Decaying epsilon for epsilon-greedy policy
        self.exploration = LinearSchedule(self.cfg.env.train_steps*0.5, 0.01)

        self.q_transformer.eval()
        self.ema_model.eval()


    def __repr__(self) -> str:
        return "Q-Transformer agent"

    def load(self):
        pass


    def get_random_actions(self):
        return torch.randint(0, self.cfg.qtransformer.num_bins, (self.cfg.env.action_dim,), device = self.device)

   
    @torch.no_grad()    
    def get_action(self, obs, step=None, eval_mode=False):

        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)

        if eval_mode:
            return self.q_transformer.get_optimal_actions(obs)

        if step < self.cfg.misc.seed_steps and not eval_mode:
            return self.q_transformer.get_random_action(obs.shape[0]).squeeze(0)
            
        # epsilon greedy
        epsilon = self.exploration.value(step)
        if random() < epsilon:
            return self.q_transformer.get_random_action(obs.shape[0]).squeeze(0)
        else:
            return self.q_transformer.get_optimal_actions(obs)
            

    
    def update(self, buffer, step):
       
        metrics = {}
        self.q_transformer.train()
        self.ema_model.train()
        self.optimizer.zero_grad()

        # sample batch
        batch = buffer.sample_batch(self.cfg.qtransformer.batch_size, self.cfg.qtransformer.n_step_td+1)
        batch = self._to_device(batch)

        loss, td_loss, conservative_reg_loss = self.q_transformer.compute_loss(batch, self.ema_model) 
        loss.backward()
        if self.cfg.qtransformer.grad_clip is not None:
            grad = nn.utils.clip_grad_norm_(self.q_transformer.parameters(), self.cfg.qtransformer.grad_clip) 
        self.optimizer.step()

        # decay lr in metaworld tasks
        if self.cfg.env.domain == "metaworld":
            self.lr_scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]

        # update target-q network
        if step % self.cfg.qtransformer.updtae_freq==0:
            ema(self.q_transformer, self.ema_model, self.cfg.qtransformer.tau)

           
        self.q_transformer.eval()
        self.ema_model.eval()

        metrics = {"loss":loss.item(), "td_loss":td_loss.item(),"conservative_loss":conservative_reg_loss.item(), "grad":grad.item(), 'lr':lr}

        return metrics




    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) for k in batch}