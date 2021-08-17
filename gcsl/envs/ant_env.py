from gcsl.envs.gymenv_wrapper import GymGoalEnvWrapper

import numpy as np
from gym import spaces
import gym 
from gcsl.envs.mujoco.ant import Env
from collections import OrderedDict


class AntGoalEnv(GymGoalEnvWrapper):
    def __init__(self, fixed_start=True, fixed_goal=True):
        self.inner_env = Env()
        super(AntGoalEnv, self).__init__(
            self.inner_env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )

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
    env = AntGoalEnv()
    # env is created, now we can use it: 
    for episode in range(1): 
        obs = env.reset()
        for step in range(5):
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)
            nobs, reward, done, info = env.step(action)
            print(env.observation(nobs).shape)
            print(env.extract_goal(nobs).shape)
