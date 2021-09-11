from gcsl.envs.ant_goal_base import AntGoalBase

import numpy as np
from gym import spaces
import gym 
from gcsl.envs.mujoco.ant import Env
from collections import OrderedDict
from gcsl.common.geometry import SkeletonGeometry
from gcsl.common.np_util import quaternion_invert, quaternion_multiply, quaternion_to_angle


class AntCurriculumGoalEnv(AntGoalBase):
    def __init__(self, fixed_start=True):
        super(AntCurriculumGoalEnv, self).__init__()
        self.skeleton = SkeletonGeometry(self.env)
        self.joint_weights = np.array([1., .5, .3, .5, .3, .5, .3, .5, .3])

        self.initial_rotation = np.array([0., 0., 1., 0.], dtype=np.float32)
        self.final_rotation = np.array([1., 0., 0., 0.], dtype=np.float32)
        self.annealing_time = 1000000
        self.total_timesteps = 0

    def update_time(self, timestep):
        self.total_timesteps = timestep

    def _reward_function(self, obs):
        nq = self.env.model.nq

        goal_xpos = self.skeleton.get_joint_locations(self.goal)
        cur_xpos = self.skeleton.get_joint_locations(obs)

        xpos_err = -40 * np.linalg.norm(goal_xpos - cur_xpos, ord=2)
        xpos_reward = np.exp(xpos_err)

        # TODO: Deepmimic use q1*q0.conj() -> then calculate the angle 
        h_diff = np.sum((obs[...,:1] - self.goal[...,:1]**2))

        q_diff = quaternion_multiply(obs[...,1:5], quaternion_invert(self.goal[...,1:5]))
        q_diff = quaternion_to_angle(q_diff)

        diff = (obs[...,5:(nq-2)] - self.goal[...,5:(nq-2)])**2   
        diff *= self.joint_weights[1:]
        diff = np.sum(diff)

        distance = h_diff + q_diff + diff
        distance_reward = np.exp(-1 * distance)

        vel_diff = (obs[...,(nq-2):] - self.goal[...,(nq-2):])**2   
        vel_diff[:6] *= self.joint_weights[0]
        vel_diff[6:] *= self.joint_weights[1:]
        vel_distance = -.2 * np.sum(vel_diff[:6])
        velocity_reward = np.exp(vel_distance)

        reward = .8 * distance_reward + .1 * velocity_reward + .1 * xpos_reward
        return reward 


    def _sample_goal(self):
        qpos = self.env.init_qpos.copy()
        # LERP initial goal and final goal
        alpha = min(self.total_timesteps / float(self.annealing_time), 1.)
        qpos[3:7] = (1 - alpha) * self.initial_rotation + alpha * self.final_rotation
        qpos[7:] = np.array([0.,  1,   0.,   -1.,   0.,   -1.,   0.,  1.])
        qvel = self.env.init_qvel.copy()
        self.goal = np.concatenate([qpos[2:], qvel])

    def _extract_z(self, goal):
        return goal[..., :1]
    
    def _extract_rotation(self, goal):
        return goal[..., 1:5]

    def _extract_q(self, goal):
        return goal[..., 5:self.env.model.nq-2]

    def _extract_qvel(self, goal):
        return goal[..., -self.env.model.nv:]

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
