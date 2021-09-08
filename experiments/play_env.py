
import gym
import numpy as np
import torch
import pathlib
import sys
sys.path.append('./')

env_name = 'ant_fixed_goal'

gpu = True
seed = 0

# Envs
from gcsl import envs
# Algo
from gcsl.algo import buffer, gcsl, variants, networks
import pathlib

if not gpu:
    print('Not using GPU. Will be slow.')

torch.manual_seed(seed)
np.random.seed(seed)

env = envs.create_env(env_name)
env_params = envs.get_env_params(env_name)

env, policy, replay_buffer, gcsl_kwargs = variants.get_params(env, env_params)

max_path_length = 500


def sample_init(render=False):
    goal_state = env.sample_goal()
    goal = env.extract_goal(goal_state)

    states = []
    actions = []

    state = env.reset()
    for t in range(max_path_length):
        state = env.reset()

        if render:
            env.render()

    

sample_init(render=True)

if __name__ == "__main__":
    params = {
        'seed': [0],
        'env_name': ['ant'], #['ant'],
        'gpu': [True],
    }
    pass
