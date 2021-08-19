"""Helper functions to create envs easily through one interface.

- create_env(env_name)
- get_goal_threshold(env_name)
"""

import numpy as np

try:
    import mujoco_py
except:
    print('MuJoCo must be installed.')

from gcsl.envs.ant_env import AntGoalEnv

env_names = ['ant']

def create_env(env_name):
    """Helper function."""
    assert env_name in env_names
    if env_name == 'ant':
        return AntGoalEnv()

def get_env_params(env_name, images=False):
    assert env_name in env_names

    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_trajectory_length=50,
        max_timesteps=1e7,
        env_name=env_name,
    )

    if env_name == 'ant':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    else:
        raise NotImplementedError()
    
    base_params.update(env_specific_params)
    return base_params