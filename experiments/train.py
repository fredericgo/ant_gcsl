from algo.gcsl import GCSL
from algo import buffer, networks 
from envs import ant_env


def get_horizon(env_params):
    return env_params.get('max_trajectory_length', 50)

def get_env_params(env_name):
    assert env_name in env_names

    base_params = dict(
        eval_freq=10000,
        eval_episodes=50,
        max_trajectory_length=50,
        max_timesteps=1e6,
    )

    if env_name == 'ant':
        env_specific_params = dict(
            goal_threshold=0.05,
        )
    else:
        raise NotImplementedError()
    
    base_params.update(env_specific_params)
    return base_params

def get_params(env, env_params):
    env = env, env_params)
    policy = default_markov_policy(env, env_params)
    buffer_kwargs = dict(
        env=env,
        max_trajectory_length=get_horizon(env_params), 
        buffer_size=20000,
    )
    replay_buffer = buffer.ReplayBuffer(**buffer_kwargs)
    gcsl_kwargs = default_gcsl_params(env, env_params)
    gcsl_kwargs['validation_buffer'] = buffer.ReplayBuffer(**buffer_kwargs)
    return env, policy, replay_buffer, gcsl_kwargs

env = ant_env.AntGoalEnv()
policy = networks.GaussianPolicy(
            env,
            state_embedding=None,
            goal_embedding=None,
            layers=[400, 300], #[400, 300], # TD3-size
            max_horizon=None, # Do not pass in horizon.
            # max_horizon=get_horizon(env_params), # Use this line if you want to include horizon into the policy
            freeze_embeddings=True,
            add_extra_conditioning=False,
        )

buffer_kwargs = dict(
        env=env,
        max_trajectory_length=get_horizon(env_params), 
        buffer_size=20000,
    )
replay_buffer = buffer.ReplayBuffer(**buffer_kwargs)

alg = GCSL(env, 
           policy,
           replay_buffer=None)
alg.train()