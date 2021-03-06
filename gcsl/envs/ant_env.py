from Cython.Compiler.ExprNodes import InnerFunctionNode
from gcsl.envs.gymenv_wrapper import GymGoalEnvWrapper

import numpy as np
from gym import spaces
import gym 
from gcsl.envs.mujoco.ant import Env
from collections import OrderedDict


class AntGoalEnv(GymGoalEnvWrapper):
    def __init__(self, fixed_start=True, fixed_goal=True):
        self.fixed_goal = True
        self.inner_env = Env()
        super(AntGoalEnv, self).__init__(
            self.inner_env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )
        

    def _sample_goal(self):
        nq = self.inner_env.model.nq
        num_jnt = self.inner_env.model.njnt
        noise = 1
        #q0 = self.inner_env.init_qpos[:7] 
        #q0[3:7] += np.random.randn(4) * .5
        q0 = self.inner_env.init_qpos[:7] + np.random.randn(7) * noise
        q0[2] = np.maximum(q0[2], 0.25)

        qb = np.random.randn(nq - 7) * noise

        qb = [np.clip(qb[i-1], 
                        self.inner_env.model.jnt_range[i][0], 
                        self.inner_env.model.jnt_range[i][1]) 
                        for i in range(1, num_jnt)]
        qpos = np.concatenate([q0, qb])
        qvel = self.inner_env.init_qvel + np.random.randn(self.inner_env.model.nv) * noise
        self.goal = np.concatenate([qpos[2:], qvel])

    def _extract_z(self, goal):
        return goal[..., :1]
    
    def _extract_rotation(self, goal):
        return goal[..., 1:5]

    def _extract_q(self, goal):
        return goal[..., 5:self.inner_env.model.nq-2]

    def _extract_qvel(self, goal):
        return goal[..., -self.inner_env.model.nv:]

    def rotation_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            qdiff = (self._extract_rotation(self.extract_goal(state)) -
                     self._extract_rotation(self.extract_goal(goal_state)))
            return np.abs(qdiff).mean(axis=-1)
            #return np.linalg.norm(qdiff, axis=-1) 
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)

    def z_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            qdiff = (self._extract_z(self.extract_goal(state)) -
                     self._extract_z(self.extract_goal(goal_state)))
            return np.abs(qdiff).mean(axis=-1)
            #return np.linalg.norm(qdiff, axis=-1) 
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)

    def goal_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            qdiff = (self.extract_goal(state) -
                     self.extract_goal(goal_state))
            return np.abs(qdiff).mean(axis=-1)
            #return np.linalg.norm(qdiff, axis=-1) 
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)

    def velocity_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            qdiff = (self._extract_qvel(self.extract_goal(state)) -
                     self._extract_qvel(self.extract_goal(goal_state)))
            return np.abs(qdiff).mean(axis=-1)
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
        rotation_distances = np.array(
            [self.rotation_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1], 1))) for i
             in range(trajectories.shape[0])])
        z_distances = np.array(
            [self.z_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1], 1))) for i
             in range(trajectories.shape[0])])
        distances = np.array(
            [self.goal_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1], 1))) for i
             in range(trajectories.shape[0])])
        vel_diff = np.array(
            [self.velocity_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1], 1))) for i
             in range(trajectories.shape[0])])
     
        return OrderedDict([
            ('mean final angle dist', np.mean(distances[:, -1])),
            ('median final angle dist', np.median(distances[:, -1])),
            ('mean angle dist', np.mean(distances)),
            ('median angle dist', np.median(distances)),
            ('mean final velocity diff', np.mean(vel_diff[:, -1])),
            ('median final velocity diff', np.median(vel_diff[:, -1])),

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
