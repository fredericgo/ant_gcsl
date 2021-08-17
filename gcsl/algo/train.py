import numpy as np

from tqdm import tqdm
from envs import ant_env

env = ant_env.AntEnv()

max_path_length = 1000
expl_noise=0.1
max_timesteps = 1000

def sample_trajectory(greedy=False, noise=0, render=False):

        goal_state = env.sample_goal()
        goal = env.extract_goal(goal_state)

        states = []
        actions = []

        state = env.reset()
        for t in range(max_path_length):
            if render:
                env.render()

            states.append(state)

            observation = env.observation(state)
            horizon = np.arange(max_path_length) >= (max_path_length - 1 - t) # Temperature encoding of horizon
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)

            #action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
            
            #if not self.is_discrete_action:
            #    action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            actions.append(action)
            state, _, _, _ = env.step(action)
        
        return np.stack(states), np.array(actions), goal_state


total_timesteps = 0

with tqdm(total=10, smoothing=0) as ranger:
    while total_timesteps < max_timesteps:
        states, actions, goal_state = sample_trajectory(greedy=True, noise=expl_noise)
        ranger.update()
        total_timesteps += max_path_length
