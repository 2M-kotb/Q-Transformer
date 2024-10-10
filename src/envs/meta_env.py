import numpy as np
import gym
from envs.wrappers import TimeLimit

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self.camera_name = "corner2"
		self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
		self.env._freeze_rand_vec = False

	def reset(self, **kwargs):
		obs = super().reset(**kwargs).astype(np.float32)
		self.env.step(np.zeros(self.env.action_space.shape))
		return obs

	def step(self, action):
		reward = 0
		for _ in range(self.cfg.action_repeat):
			obs, r, _, info = self.env.step(action.copy())
			reward += r
		obs = obs.astype(np.float32)
		return obs, reward, False, info

	@property
	def unwrapped(self):
		return self.env.unwrapped

	def render(self, *args, **kwargs):
		return self.env.render(
			offscreen=True, resolution=(384, 384), camera_name=self.camera_name
		).copy()


def make_env(cfg):
	"""
	Make Meta-World environment.
	"""
	env_id = cfg.task.split("-", 1)[-1] + "-v2-goal-observable"
	if not cfg.task.startswith('mw-') or env_id not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.modality == 'state', 'This task only supports state observations.'
	cfg.episode_length = 200//cfg.action_repeat
	cfg.train_steps = cfg.train_steps//cfg.action_repeat
	env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=cfg.seed)
	env = MetaWorldWrapper(env, cfg)
	env = TimeLimit(env, max_episode_steps=cfg.episode_length)
	env.max_episode_steps = env._max_episode_steps
	cfg.obs_shape = tuple(int(x) for x in env.observation_space.shape)
	action_shape = tuple(int(x) for x in env.action_space.shape)
	cfg.action_dim = env.action_space.shape[0]
	cfg.scale = 0.5
	return env
