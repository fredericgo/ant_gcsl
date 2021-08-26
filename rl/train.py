import argparse
import datetime
import gym
import numpy as np
import itertools
import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from rl.sac import SAC
from rl.buffer import ReplayBuffer
from gcsl.envs import create_env

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="ant_fixed",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=4000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--max_trajectory_length', type=int, default=50, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=10, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--reward_scaling', type=int, default=10, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--checkpoint_interval', type=int, default=1000, 
                    help='checkpoint training model every # steps')
parser.add_argument('--log_interval', type=int, default=10, 
                    help='checkpoint training model every # steps')
parser.add_argument('--eval_interval', type=int, default=1000, 
                    help='checkpoint training model every # steps')

args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = create_env(args.env_name)

print(env)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Agent
agent = SAC(env, args)

#Tesnorboard
datetime_st = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/{datetime_st}_SAC_{args.env_name}_{args.policy}'
writer = SummaryWriter(log_dir)


# Memory
memory = ReplayBuffer(env, args.max_trajectory_length, args.replay_size)

def sample_trajectory(env, greedy=False, noise=0, render=False):
    goal_state = env.sample_goal()
    goal = env.extract_goal(goal_state)

    states = []
    actions = []
    next_states = []
    rewards = []

    done = False
    state = env.reset()
    for t in range(args.max_trajectory_length):
        if render:
            env.render()

        states.append(state)
        if done:
            state = env.reset()

        observation = env.observation(state)
        horizon = np.arange(args.max_trajectory_length) >= (args.max_trajectory_length - 1 - t) # Temperature encoding of horizon horizon[None],
        action = agent.select_action(observation[None], goal[None])[0]
        
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards.append(reward)
        next_states.append(next_state)
    return np.stack(states), np.array(actions), np.stack(next_states), goal_state, np.stack(rewards)


def evaluate_policy(env, eval_episodes=10, greedy=True, prefix='Eval', total_timesteps=0):
    
    all_states = []
    all_goal_states = []
    all_actions = []
    final_dist_vec = np.zeros(eval_episodes)
    success_vec = np.zeros(eval_episodes)

    for index in range(eval_episodes):
        states, actions, _, goal_state, rewards = sample_trajectory(env, noise=0, greedy=greedy, render=False)
        all_actions.extend(actions)
        all_states.append(states)
        all_goal_states.append(goal_state)
        final_dist = env.goal_distance(states[-1], goal_state)
        final_dist_vec[index] = final_dist
        success_vec[index] = (final_dist < 0.05)
    all_states = np.stack(all_states)
    all_goal_states = np.stack(all_goal_states)
    print('%s num episodes'%prefix, eval_episodes)
    print('%s avg final dist'%prefix,  np.mean(final_dist_vec))
    print('%s success ratio'%prefix, np.mean(success_vec))
    print('%s reward'%prefix, np.mean(rewards))

    writer.add_scalar('%s/avg final dist'%prefix, np.mean(final_dist_vec), total_timesteps)
    writer.add_scalar('%s/success ratio'%prefix,  np.mean(success_vec), total_timesteps)
    writer.add_scalar('%s/reward'%prefix,  np.mean(rewards), total_timesteps)
    
    diagnostics = env.get_diagnostics(all_states, all_goal_states)
    for key, value in diagnostics.items():
        print('%s %s'%(prefix, key), value)
    return all_states, all_goal_states
# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    states, actions, next_states, goal_state, _ =  sample_trajectory(env)
    memory.add_trajectory(states, actions, next_states, goal_state) # Append transition to memory

    for i in range(args.updates_per_step):
        # Update parameters of all the networks
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
        writer.add_scalar('loss/critic_1', critic_1_loss, updates)
        writer.add_scalar('loss/critic_2', critic_2_loss, updates)
        writer.add_scalar('loss/policy', policy_loss, updates)
        writer.add_scalar('loss/entropy_loss', ent_loss, updates)
        writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        updates += 1
    

    if total_numsteps > args.num_steps:
        break
        
    if i_episode % args.eval_interval == 0 and args.eval is True:
        evaluate_policy(env, total_timesteps=updates)

    if i_episode % args.checkpoint_interval == 0:
        agent.save_model(log_dir)
        print("----------------------------------------")
        print(f"Save Model: {i_episode} episodes.")
        print("----------------------------------------")

env.close()

