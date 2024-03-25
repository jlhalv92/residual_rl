import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.algorithm.deep_actor_critic.transfer_rl import TransferRL
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from src.envs.dm_control_env import DMControl
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from src.networks.networks import CriticNetwork, ActorNetwork, TransferCritic
from tqdm import trange
import wandb


def experiment(alg, n_epochs, n_steps, n_episodes_test, run_id):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    horizon = 500
    gamma = 0.99
    mdp = DMControl('walker', 'cargo', horizon, gamma, use_pixels=False)
    # mdp.env.task._move_speed = 4.

    wandb.init(
        # set the wandb project where this run will be logged
        project="walker_cargo_comparison",
        name="extended_transfer_v2"
    )

    # Settings
    initial_replay_size = 0.01*n_steps*n_epochs
    max_replay_size = n_steps*n_epochs
    batch_size = 400
    n_features = 400
    warmup_transitions = 0.015*n_steps*n_epochs
    tau = 0.005
    lr_alpha = 3e-4

    use_cuda = torch.cuda.is_available()

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': 3e-4}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=TransferCritic,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': 3e-4}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                log_std_min=-3, log_std_max=2, target_entropy=None, critic_fit_params=None)

    # Algorithm

    core = Core(agent, mdp)

    old_agent = alg.load("src/nominal_models/walker/nominal_walker")

    agent.setup_transfer(prior_agents=[old_agent], use_policy=False, unfreeze_weights=False)

    # RUN
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)

    logger.epoch_info(0, J=J, R=R, entropy=E)

    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):
        core.learn(n_steps=n_steps, n_steps_per_fit=1)
        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        logger.epoch_info(n+1, J=J, R=R, entropy=E)
        logs_dict = {"RETURN": J, "REWARD": R}
        wandb.log(logs_dict, step=n)

    # logger.info('Press a button to visualize pendulum')
    # input()
    # core.evaluate(n_episodes=5, render=True)
    agent.save("checkpoint/extended_cargo_v2{}".format(run_id))
    wandb.finish()

if __name__ == '__main__':
    algs = [
        TransferRL
    ]

    for i in range(5):
        experiment(alg=TransferRL, n_epochs=20, n_steps=5000, n_episodes_test=10, run_id=i)