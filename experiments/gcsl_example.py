
import gym
import numpy as np
import torch

output_dir = '/tmp', 
env_name = 'ant_fixed'
gpu = True
seed = 0

# Envs
from gcsl import envs
# Algo
from gcsl.algo import buffer, gcsl, variants, networks

if not gpu:
    print('Not using GPU. Will be slow.')

torch.manual_seed(seed)
np.random.seed(seed)

env = envs.create_env(env_name)
env_params = envs.get_env_params(env_name)
print(env_params)

env, policy, replay_buffer, gcsl_kwargs = variants.get_params(env, env_params)
algo = gcsl.GCSL(
    env,
    policy,
    replay_buffer,
    **gcsl_kwargs
)

algo.train()


if __name__ == "__main__":
    params = {
        'seed': [0],
        'env_name': ['pointmass_empty'], #['ant'],
        'gpu': [True],
    }
    pass
