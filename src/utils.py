import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble
import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from einops import pack, unpack, repeat, reduce, rearrange




#-------------------------------------------------------------------------------
#================
# schedule epsilon
#=================
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


#-------------------------------------------------------------------------------
#==============
# MISC
#==============


def extract_state_dict(state_dict, module_name):
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)

def ema(m, m_target, tau):
	"""Update slow-moving average of online network (target network) at rate tau."""
	with torch.no_grad():
		for p, p_target in zip(m.parameters(), m_target.parameters()):
			p_target.data.lerp_(p.data, tau)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def batch_select_indices(t, indices):
    if t.ndim == 4:
        indices = repeat(indices, 'b n -> b n 1 d', d = t.shape[-1])
        selected = t.gather(-2, indices)
        return selected.squeeze(-2)
        # indices = rearrange(indices, '... -> ... 1 1')
        # indices = indices.repeat(1,1,1,t.shape[-1])
        # selected = t.gather(-2, indices)
        # return selected.squeeze(-2)

    indices = rearrange(indices, '... -> ... 1')
    selected = t.gather(-1, indices)
    return rearrange(selected, '... 1 -> ...')


#-------------------------------------------------------------------------------
#==============
# Replay Buffer
#==============


Batch = Dict[str, torch.Tensor]

@dataclass
class Segment:
    observations: torch.ByteTensor
    actions: torch.FloatTensor
    rewards: torch.FloatTensor
    dones: torch.bool
    return_to_go: torch.FloatTensor


class Episode(object):
    """Storage object for a single episode."""
    def __init__(self, cfg, init_obs):
        self.cfg = cfg
        device =  'cpu'
        self.device = torch.device(device)
        self.obs_dtype = torch.float32 if cfg.modality == 'state' else torch.uint8
        self.obs = []
        self.obs.append(torch.tensor(init_obs, dtype=self.obs_dtype))
        self.action = []
        self.reward = []
        self.dones = []
        self.return_to_go = []
        
        self.cumulative_reward = 0
        self.G = 0
        self.discount = 1
        self.done = False
        self.success = 0
        self._idx = 0

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def reward_scale(self, reward):
        return (reward * self.cfg.scale)
        #return symlog(torch.tensor(reward, dtype=torch.float32))

    def add(self, obs, action, reward, done, success):

        if not done:
            # ignore terminal observation
            self.obs.append(torch.tensor(obs, dtype=self.obs_dtype))
        self.action.append(torch.tensor(action, dtype=torch.int64))
        self.reward.append(torch.tensor(self.reward_scale(reward), dtype=torch.float32))
        self.dones.append(torch.tensor(done, dtype=torch.bool))
        self.cumulative_reward += reward
        self.G += self.discount * self.reward_scale(reward)
        self.discount *= self.cfg.discount
        self.done = done
        self.success = int(success)
        self._idx += 1
        if done:
            self.compute_return_to_go()
            self.list_to_tensor()

    def compute_return_to_go(self):
        #the 1st time step has the entire return 
        self.return_to_go.append(torch.tensor(self.G, dtype=torch.float32))
        R = self.G
        for r in self.reward[:-1]:
            R = (R - r)/self.cfg.discount
            self.return_to_go.append(torch.tensor(R, dtype=torch.float32))


    def list_to_tensor(self):
        self.obs = torch.stack(self.obs).to(self.device)
        self.action = torch.stack(self.action).to(self.device)
        self.reward = torch.stack(self.reward).to(self.device)
        self.dones = torch.stack(self.dones).to(self.device)
        self.return_to_go = torch.stack(self.return_to_go).to(self.device)

    def segment(self,start: int, stop: int)-> Segment:
        assert start < len(self) and stop > 0 and start < stop
       
        observations = self.obs[start:stop]
        actions = self.action[start:stop]
        rewards = self.reward[start:stop]
        dones = self.dones[start:stop]
        return_to_go = self.return_to_go[start:stop]

        return Segment(observations, actions, rewards, dones, return_to_go)

class ReplayBuffer():
    """

    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_num_episodes = cfg.max_num_episodes
        self.num_seen_episodes = 0
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()
        self.weights = deque()

    def __len__(self) -> int:
        return len(self.episodes)

    def __add__(self, episode: Episode):
        _ = self.add_episode(episode)
        return self

    def add_episode(self, episode: Episode) -> int:
        if self.max_num_episodes is not None and len(self.episodes) == self.max_num_episodes:
            self._popleft()
            self.weights.popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def _append_new_episode(self, episode):
        episode_id = self.num_seen_episodes
        self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
        self.episodes.append(episode)
        if episode.cumulative_reward == 0.0:
            self.weights.append(episode.cumulative_reward+1e-4)
        else:
            self.weights.append(episode.cumulative_reward)
        self.num_seen_episodes += 1
        return episode_id

    def _popleft(self) -> Episode:
        id_to_delete = [k for k, v in self.episode_id_to_queue_idx.items() if v == 0]
        assert len(id_to_delete) == 1
        self.episode_id_to_queue_idx = {k: v - 1 for k, v in self.episode_id_to_queue_idx.items() if v > 0}
        return self.episodes.popleft()


    def sample_batch(self, batch_num_samples: int, sequence_length: int) -> Batch:
        return self._collate_episodes_segments(self._sample_episodes_segments(batch_num_samples, sequence_length))

    def _sample_episodes_segments(self, batch_num_samples: int, sequence_length: int) -> List[Segment]:
        sampled_episodes = random.choices(self.episodes, weights = list(self.weights), k=batch_num_samples) #
        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            start = random.randint(0, len(sampled_episode) - (sequence_length + 1))
            stop = start + sequence_length
            sampled_episodes_segments.append(sampled_episode.segment(start, stop))
            assert sampled_episodes_segments[-1].observations.size(0) == sequence_length
        return sampled_episodes_segments

    def _collate_episodes_segments(self, episodes_segments: List[Segment]) -> Batch:
        episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        batch = {}
        for k in episodes_segments[0]:
            batch[k] = torch.stack([e_s[k] for e_s in episodes_segments])
        batch['observations'] = batch['observations'].float() / 255.0 if self.cfg.modality=='pixels' else batch['observations']
        return batch

