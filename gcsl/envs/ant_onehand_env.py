from gcsl.envs.gymenv_wrapper import GymGoalEnvWrapper

import numpy as np
from gym import spaces
import gym 
from gcsl.envs.mujoco.ant_onehand import Env
from collections import OrderedDict


class AntOnehandGoalEnv(GymGoalEnvWrapper):
    def __init__(self, fixed_start=True, fixed_goal=True):
        self.inner_env = Env()
        super(AntOnehandGoalEnv, self).__init__(
            self.inner_env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )
        

    def _sample_goal(self):
        qpos = np.array([0.,  1,   0.,   -1.,   0.,   -1.,   0.,  1.])
        qpos[1] = np.clip(np.random.randn(), 
                          self.inner_env.joint_range[0], 
                          self.inner_env.joint_range[1])

        qvel = self.inner_env.init_qvel
        self.goal = np.concatenate([qpos, qvel])

    def goal_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            qdiff = (self.extract_goal(state) -
                     self.extract_goal(goal_state)) 
            return np.abs(qdiff)[1]
            #return np.linalg.norm(qdiff, axis=-1) 
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)


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
     
        return OrderedDict([
            ('mean final angle dist', np.mean(distances[:, -1])),
            ('median final angle dist', np.median(distances[:, -1])),
        ])


if __name__ == "__main__":
    env = AntGoalEnv()
    # env is created, now we can use it: 
    for episode in range(1): 
        obs = env.reset()
        for step in range(5):
            action = env.action_space.sample()  # or given a custom model, action = policy(observation)
            nobs, reward, done, info = env.step(action)
            print(env.observation(nobs))
            print(env.extract_goal(nobs).shape)
