from copy import deepcopy
import warnings

import gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
    raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
    from envs.dmcontrol import make_env as make_dm_control_env
except:
    make_dm_control_env = missing_dependencies
try:
    from envs.maniskill import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies
try:
    from envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from envs.myosuite import make_env as make_myosuite_env
except:
    make_myosuite_env = missing_dependencies
try: 
    from envs.languagetable import make_env as make_lt_env
except:
    make_lt_env = missing_dependencies
try: 
    from envs.frankakitchen import make_env as make_fk_env
except: 
    make_fk_env = missing_dependencies
try:
    from envs.simplerenv import make_env as make_simpler_env 
except: 
    make_simpler_env = missing_dependencies


print('the simpler env is', make_simpler_env)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
    """
    Make a multi-task environment for TD-MPC2 experiments.
    """
    print('Creating multi-task environment with tasks:', cfg.tasks)
    envs = []
    for task in cfg.tasks:
        
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError('Unknown task:', task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env
    

def make_env(cfg):
    """
    Make an environment for TD-MPC2 experiments.
    """
    gym.logger.set_level(40)
    if cfg.multitask:
        env = make_multitask_env(cfg)

    else:
        env = None
        for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_lt_env, make_fk_env, make_simpler_env]:
            try:
                env = fn(cfg)
            except ValueError:
                pass
        if env is None:
            raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
        env = TensorWrapper(env)
    print('----- obs is ', cfg.get('obs', 'state'))
    if cfg.get('obs', 'state') == 'rgb':
        env = PixelWrapper(cfg, env, num_frames=1)
    try: # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except: # Box
        cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5*cfg.episode_length) if cfg.seed_steps is  None else cfg.seed_steps
    return env
