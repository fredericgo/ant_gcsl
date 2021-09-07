from gcsl.envs import goal_env
from gcsl.envs.mujoco.ant import Env

import numpy as np
import gym
from gym.spaces import Box


class AntRootGoalEnv(goal_env.GoalEnv):
    
    """
    
    A wrapper around mujoco environments. and gym GoalEnvs

    """
    
    def __init__(self):
        super(AntRootGoalEnv, self).__init__()
        self.env = Env()

        self.action_space = self.env.action_space

        self.observation_space = self.env.observation_space
        #self.goal_space = self.env.observation_space
        self.goal_space = Box(low=-np.inf, high=np.inf, shape=(5,))        
        self.sgoal_space = Box(low=-np.inf, high=np.inf, shape=(5,))   # achieved state ~ goal

        # concat observation and goal to get the state
        obs_low = self.observation_space.low.flatten()
        goal_low = self.goal_space.low.flatten()
        sgoal_low = self.sgoal_space.low.flatten()
        state_low = np.r_[obs_low, goal_low, sgoal_low]

        obs_hi = self.observation_space.high.flatten()
        goal_hi = self.goal_space.high.flatten()
        sgoal_hi = self.sgoal_space.high.flatten()
        state_high = np.r_[obs_hi, goal_hi, sgoal_hi]

        self.state_space = gym.spaces.Box(low=state_low, high=state_high)

        self.obs_dims = obs_low.shape[0]
        self.goal_dims = goal_low.shape[0]
        self.sgoal_dims = sgoal_low.shape[0]
        self.goal = np.zeros(self.goal_dims)

    def _base_obs_to_state(self, base_obs):
        obs = base_obs.flatten()
        goal = base_obs[..., :5].flatten()
        sgoal = base_obs.flatten()
        return np.r_[obs, goal, sgoal]

    def _sample_goal(self):
        qpos = self.env.init_qpos.copy()[:7]
        #qpos[7:] = np.array([0.,  1,   0.,   -1.,   0.,   -1.,   0.,  1.])
        qvel = self.env.init_qvel.copy()
        self.goal = qpos

    def reset(self):
        """
        Resets the environment and returns a state vector

        Returns:
            The initial state
        """
        base_obs = self.env.reset()
        return self._base_obs_to_state(base_obs)

    def render(self, mode='human'):
        return self.env.render(mode=mode)
        
    def step(self, a):
        """
        Runs 1 step of simulation

        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """
        ns, reward, done, infos = self.env.step(a)
        infos['observation'] = ns

        reward = self._reward_function(ns)
        ns = self._base_obs_to_state(ns)
        return ns, reward, done, infos

    def _reward_function(self, obs):
        nq = self.env.model.nq
        distance =  - np.linalg.norm(obs[...,:5] - self.goal[...,:5], ord=2)
        distance_reward = np.exp(distance)
        reward = 1 * distance_reward
        return reward 

    def observation(self, state):
        """
        Returns the observation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        obs = state[...,:self.obs_dims]
        return obs.reshape(obs.shape[:len(obs.shape)-1]+self.observation_space.shape)
    
    def extract_goal(self, state):
        """
        Returns the goal representation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        goal = state[...,self.obs_dims:self.obs_dims+self.goal_dims]
        return goal.reshape(goal.shape[:len(goal.shape)-1]+self.goal_space.shape)

    def _extract_sgoal(self, state):
        """
        Returns the state goal representation for a given state (internal)

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        sgoal = state[..., self.obs_dims + self.goal_dims:]
        return sgoal.reshape(sgoal.shape[:len(sgoal.shape)-1]+self.sgoal_space.shape)

    def sample_goal(self):
        """
        Samples a goal state (of type self.state_space.sample()) using 'desired_goal'
        
        """
        self._sample_goal()
        obs = (10 + self.observation_space.sample()).flatten() # Placeholder - shouldn't actually be used
        goal = self.goal.flatten()
        sgoal = self.goal.flatten()
        return np.r_[obs, goal, sgoal]
    
    def goal_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            diff = self.extract_goal(state) - self.extract_goal(goal_state)
            return np.linalg.norm(diff, axis=-1) 
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)
    
    def set_state(self, qpos, qvel):
        self.env.set_state(qpos, qvel)
