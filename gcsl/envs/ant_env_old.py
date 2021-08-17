from envs.goal_env import GoalEnv

import numpy as np
from gym import spaces
import gym 
from envs.mujoco.ant import Env
from collections import OrderedDict

class AntEnv(GoalEnv):
    def __init__(self, fixed_start=True, fixed_goal=True):
        self.inner_env = Env()
        # state space
        self.state_space = self.inner_env.observation_space
        self.observation_space = self.inner_env.observation_space
        self.goal_space = self.inner_env.observation_space

        # action space
        self.action_space = self.inner_env.action_space
        self.state_dim = 27

    def reset(self):
        state = self.inner_env.reset()
        return state

    def step(self, action):
        state, reward, done, info = self.inner_env.step(action)
        return state, 0, False, info

    def observation(self, state):
        return state

    def extract_goal(self, state):
        return state[..., -self.state_dim:]

    def set_to_goal(self, goal):
        self.inner_env.set_to_goal(goal)

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Gets things to log

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """

        distances = np.array(
            [self.goal_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1], 1))) for i
             in range(trajectories.shape[0])])
        amount_moved = np.array(
            [self.goal_distance(trajectories[i], np.tile(trajectories[i][0], (trajectories.shape[1], 1))) for i
             in range(trajectories.shape[0])])

        return OrderedDict([
            ('mean final angle dist', np.mean(distances[:, -1])),
            ('median final angle dist', np.median(distances[:, -1])),
            ('mean final angle moved', np.mean(amount_moved[:, -1])),
            ('median final angle moved', np.median(amount_moved[:, -1])),

        ])


if __name__ == "__main__":
    env = AntEnv()
    # env is created, now we can use it: 
    g = np.zeros(27)
    env.set_to_goal(g)
    print(env.observation_space)
    for episode in range(1): 
        obs = env.reset()
        for step in range(5):
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)
            nobs, reward, done, info = env.step(action)
            print(nobs.shape)
            print(env.extract_goal(nobs))
