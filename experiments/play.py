
import gym
import numpy as np
import torch
import pathlib
import sys
sys.path.append('./')

output_dir = '/tmp', 
env_name = 'ant_onehand'

model_dir = 'runs/ant_onehand_2021-08-23_16-00-45'
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


def load_policy(model_dir):
    policy_path = pathlib.Path(model_dir) / "policy.pkl"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))


def sample_init(greedy=False, noise=0, render=False):
    goal_state = env.sample_goal()
    goal = env.extract_goal(goal_state)

    states = []
    actions = []

    state = env.reset()
    for t in range(max_path_length):
        state = env.reset()

        if render:
            env.render()

        

def sample_trajectory(greedy=False, noise=0, render=False):
    goal_state = env.sample_goal()
    goal = env.extract_goal(goal_state)
    qpos = goal[:env.inner_env.model.nq]
    qvel = goal[env.inner_env.model.nq:]
    env.set_state(qpos, qvel)

    for _ in range(100):
        env.render()
    
    states = []
    actions = []

    state = env.reset()
    done = False
    for t in range(max_path_length):
        if render:
            env.render()

        states.append(state)
        if done:
            state = env.reset()

        observation = env.observation(state)
        horizon = np.arange(max_path_length) >= (max_path_length - 1 - t) # Temperature encoding of horizon horizon[None],
        action = policy.act_vectorized(observation[None], goal[None], horizon=None, greedy=greedy, noise=noise)[0]
        
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        actions.append(action)
        state, _, done, _ = env.step(action)
       
    return np.stack(states), np.array(actions), goal_state

load_policy(model_dir)
sample_trajectory(render=True)
sample_trajectory(render=True)

#sample_init(noise=1, render=True)

if __name__ == "__main__":
    params = {
        'seed': [0],
        'env_name': ['ant'], #['ant'],
        'gpu': [True],
    }
    pass
