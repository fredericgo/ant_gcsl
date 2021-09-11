
import gym
import numpy as np
import torch
import pathlib
import sys
sys.path.append('./')

import imageio
from rl.networks import GaussianPolicy

output_dir = '/tmp', 
env_name = 'ant_curriculum_goal'

model_dir = 'runs/2021-09-09_15-15-02_SAC_ant_fixed_goal_Gaussian'
video_file = 'video.mp4'
hidden_size = 512
gpu = True
interactive = False if video_file else True
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

action_dim = env.action_space.shape[0]
policy = GaussianPolicy(env.obs_dims, action_dim, env.goal_dims, hidden_size, env.action_space)


max_path_length = 50


def load_policy(model_dir):
    policy_path = pathlib.Path(model_dir) / "models" / "actor"
    policy.load_state_dict(torch.load(policy_path, map_location=torch.device('cpu')))

def select_action(policy, state, goal, evaluate=False):
    state = torch.FloatTensor(state).unsqueeze(0)
    goal = torch.FloatTensor(goal).unsqueeze(0)

    if evaluate is False:
        action, _, _ = policy.sample(state, goal)
    else:
        _, _, action = policy.sample(state, goal)
    return action.detach().cpu().numpy()[0]

def sample_trajectory(writer, greedy=False, noise=0):
    goal_state = env.sample_goal()
    goal = env.extract_goal(goal_state)
    qpos = np.concatenate([[0, 0],goal[:(env.env.model.nq-2)]])
    qvel = goal[(env.env.model.nq-2):]
    env.set_state(qpos, qvel)

    for _ in range(30):
        if interactive:
            env.render()
        else:
            writer.append_data(env.render(mode="rgb_array"))
    
    states = []
    actions = []

    state = env.reset()
    done = False
    for t in range(max_path_length):
        if interactive:
            env.render()
        else:
            writer.append_data(env.render(mode="rgb_array"))
        states.append(state)
        if done:
            state = env.reset()

        observation = env.observation(state)
        horizon = np.arange(max_path_length) >= (max_path_length - 1 - t) # Temperature encoding of horizon horizon[None],
        action = select_action(policy, observation, goal)
        
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        actions.append(action)
        state, r, done, _ = env.step(action)
       
    env.update_time(t+1)
    return np.stack(states), np.array(actions), goal_state

load_policy(model_dir)

writer = imageio.get_writer(video_file, fps=30) 
for _ in range(20):
    sample_trajectory(writer)

#sample_init(noise=1, render=True)

if __name__ == "__main__":
    params = {
        'seed': [0],
        'env_name': ['ant'], #['ant'],
        'gpu': [True],
    }
    pass
