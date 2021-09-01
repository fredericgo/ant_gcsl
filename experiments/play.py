
import gym
import numpy as np
import torch
import pathlib
import sys
sys.path.append('./')

import imageio

output_dir = '/tmp', 
env_name = 'reacher_goal'

model_dir = 'runs/ant_fixed_goal_2021-08-30_13-58-51'
video_file = 'video.mp4'
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

max_path_length = 50


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

        

def sample_trajectory(writer, greedy=False, noise=0):
    goal_state = env.sample_goal()
    goal = env.extract_goal(goal_state)
    qpos = np.concatenate([[0, 0],goal[:(env.inner_env.model.nq-2)]])
    qvel = goal[(env.inner_env.model.nq-2):]
    env.set_state(qpos, qvel)

    for _ in range(100):
        writer.append_data(env.render(mode="rgb_array"))

    
    states = []
    actions = []

    state = env.reset()
    done = False
    for t in range(max_path_length):
        writer.append_data(env.render(mode="rgb_array"))

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

writer = imageio.get_writer(video_file, fps=30) 
for _ in range(10):
    sample_trajectory(writer)

#sample_init(noise=1, render=True)

if __name__ == "__main__":
    params = {
        'seed': [0],
        'env_name': ['ant'], #['ant'],
        'gpu': [True],
    }
    pass
