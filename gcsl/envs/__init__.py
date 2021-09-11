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
from gcsl.envs.ant_onehand_env import AntOnehandGoalEnv
from gcsl.envs.ant_fixed_env import AntFixedGoalEnv
from gcsl.envs.ant_z_env import AntZGoalEnv
from gcsl.envs.ant_fixed_goal_env import AntFixedGoalEnv
from gcsl.envs.reacher_goal_env import ReacherGoalEnv
from gcsl.envs.ant_root_goal_env import AntRootGoalEnv
from gcsl.envs.ant_curriculum_goal_env import AntCurriculumGoalEnv


env_names = ['ant', 'ant_onehand', 'ant_fixed', 
             'ant_z', 'ant_fixed_goal', 'reacher_goal',
             'ant_root_goal', 'ant_curriculum_goal']

def create_env(env_name):
    """Helper function."""
    assert env_name in env_names
    if env_name == 'ant':
        return AntGoalEnv()
    elif env_name == 'ant_onehand':
        return AntOnehandGoalEnv()
    elif env_name == 'ant_fixed':
        return AntFixedGoalEnv()
    elif env_name == 'ant_z':
        return AntZGoalEnv()
    elif env_name == 'ant_fixed_goal':
        return AntFixedGoalEnv()
    elif env_name == 'reacher_goal':
        return ReacherGoalEnv()
    elif env_name == 'ant_root_goal':
        return AntRootGoalEnv()
    elif env_name == 'ant_curriculum_goal':
        return AntCurriculumGoalEnv()

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
    elif env_name == 'ant_onehand':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ant_fixed':
        env_specific_params = dict(
            goal_threshold=0.05,
            max_path_length=50,
        )
    elif env_name == 'ant_z':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ant_fixed_goal':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'ant_root_goal':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    elif env_name == 'reacher_goal':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    else:
        raise NotImplementedError()
    
    base_params.update(env_specific_params)
    return base_params