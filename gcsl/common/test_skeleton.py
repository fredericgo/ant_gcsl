
from gym.core import GoalEnv
from gcsl.envs import create_env
from gcsl.common.skeleton import Skeleton
import torch

env = create_env("ant_fixed_goal")
sk = Skeleton(env.env)

goal = env.sample_goal()
goal = env.extract_goal(goal)
goal = torch.as_tensor(goal)

x = sk.get_joint_locations(goal)

print(x)