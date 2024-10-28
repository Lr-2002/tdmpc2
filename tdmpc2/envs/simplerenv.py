import numpy as np
from absl import app
from PIL import Image
from envs.wrappers.time_limit import TimeLimit
# from wrappers.time_limit import TimeLimit
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

from omegaconf import OmegaConf
import gymnasium as gym 
class SEWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape,
                        self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape,
                         self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )

    def get_img(self, obs):
        self.obs = obs 
        return get_image_from_maniskill2_obs_dict(self.env, obs)

    def reset(self):
        obs, reset_info = self.env.reset()
        rgb = self.get_img(obs)  
        assert np.min(rgb) >= 0 
        return rgb

    def step(self, action):
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs['rgb'] = rgb
        obs = self.get_img(obs)

        return (obs, reward, terminated or truncated, info)
    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, args, **kwargs):
        return self.get_img(self.obs)

def make_env(cfg):
    if 'simpler' not in cfg.task: # franka kitchen
        raise ValueError('no such task in language table ')


    assert cfg.get('obs', 'state') == 'rgb', 'FrankaKitchen only support for rgb obs '
    env = simpler_env.make('google_robot_pick_coke_can')
 
    env = SEWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)
    env.max_episode_steps = env._max_episode_steps
    return env


if __name__ == '__main__':
    conf = OmegaConf.create({'obs': 'rgb', 'task': 'simpler', 'episode_length':100, 'task': 'simpler-test'})
    env = make_env(conf)
    obs = env.reset()
    done, truncated = False, False
    while not (done or truncated):
       # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
       # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
       action = env.action_space.sample() # replace this with your policy inference
       obs, reward, done, info = env.step(action) # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold
       print(obs.shape)


