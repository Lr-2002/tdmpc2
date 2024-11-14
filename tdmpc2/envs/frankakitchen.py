import numpy as np
# from envs.wrappers.time_limit import TimeLimit
from absl import app
from PIL import Image
from envs.wrappers.time_limit import TimeLimit
import gymnasium as gym 
import gymnasium_robotics 
gym.register_envs(gymnasium_robotics)
class FKWrapper(gym.Wrapper):
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

    def get_img(self):
        return self.env.render().copy()

    def reset(self):
        self.env.reset()
        rgb = self.get_img()  
        assert np.min(rgb) >= 0 
        return rgb

    def step(self, action):
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs['rgb'] = rgb
        obs = self.get_img()

        return (obs, reward, terminated or truncated, info)
    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, args, **kwargs):
        return self.env.render()


def make_env(cfg):
    if 'fk-' not in cfg.task: # franka kitchen
        raise ValueError('no such task in language table ')


    assert cfg.get('obs', 'state') == 'rgb', 'FrankaKitchen only support for rgb obs '

    # print('init the task', cfg.task)
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave','kettle','hinge cabinet','slide cabinet','light switch','top burner','bottom burner'], render_mode='rgb_array')
    env = FKWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)
    env.max_episode_steps = env._max_episode_steps
    return env


if __name__ == '__main__':
    print('running the code ')
    env = make_env()
    env.reset()
    res = env.render()
    print(env.action_space)
    action = np.random.rand(9)
    print(action)

    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    img = Image.fromarray(res) 
    img.save('test.jpg')
