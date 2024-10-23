# coding=utf-8
# Copyright 2024 The Language Tale Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example for running the Language-Table environment."""

from language_table.environments.rewards import block2block_relative_location
from language_table.environments import language_table
from language_table.environments import blocks
from collections.abc import Sequence
import numpy as np
from envs.wrappers.time_limit import TimeLimit
from absl import app
import gym  # im
# from matplotlib import pyplot as plt
#
#
# def main(argv):
#
#   env = language_table.LanguageTable(
#       block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
#       reward_factory=block2block.BlockToBlockReward,
#       control_frequency=10.0,
#   )
#   _ = env.reset()
#
#   # Take a few random actions.
#   for _ in range(5):
#     env.step(env.action_space.sample())
#
#   # Save a rendered image.
#   plt.imsave('/tmp/language_table_render.png', env.render())
#BlockToBlockRelativeLocationReward


class LTWrapper(gym.Wrapper):
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

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, args, **kwargs):
        return self.env.render()


def make_env(cfg):
    if 'lt' not in cfg.task:
        raise ValueError('no such task in language table ')

    print('init the task', cfg.task)
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=block2block_relative_location.BlockToBlockRelativeLocationReward,
        control_frequency=10.0,
    )
    env = LTWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)
    env.max_episode_steps = env._max_episode_steps
    return env


if __name__ == '__main__':
    print('running the code ')
    env = make_env()
    print(env, env.action_space, env.observation_space)
