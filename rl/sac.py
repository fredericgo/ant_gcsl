import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from rl.utils import soft_update, hard_update
from rl.networks import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, env, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.batch_size = args.batch_size

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        self.reward_scaling = args.reward_scaling
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.goal_dim = env.goal_space.shape[0]
        self.critic = QNetwork(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size, env.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(self.state_dim, self.action_dim, self.goal_dim, args.hidden_size, env.action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, goal, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.policy.sample(state, goal)
        else:
            _, _, action = self.policy.sample(state, goal)
        return action.detach().cpu().numpy()[0]

    def reward_function(self, state, goal):
        nq = self.env.inner_env.model.nq
        distance = -2 * torch.sum((state[...,:nq-2] - goal[...,:nq-2])**2, -1)
        distance_reward = torch.exp(distance)
        velocity_diff = -.5 * torch.sum((state[...,(nq-2):] - goal[...,(nq-2):])**2, -1)
        velocity_reward = torch.exp(velocity_diff)
        return distance_reward + 0.1 * velocity_reward

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        #state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch, action_batch, next_state_batch, goal_batch, _, _, _ = memory.sample_batch(self.batch_size)            
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        goal_batch = torch.FloatTensor(goal_batch).to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch, goal_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action, goal_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            reward = self.reward_scaling * self.reward_function(state_batch, goal_batch).unsqueeze(-1)
            next_q_value = reward + self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch, goal_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch, goal_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi, goal_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, dir_name):
        model_path = os.path.join(dir_name, 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        actor_path = os.path.join(model_path, 'actor')
        critic_path = os.path.join(model_path, 'critic')
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, model_path):
        actor_path = os.path.join(model_path, 'actor')
        critic_path = os.path.join(model_path, 'critic')

        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

