import numpy as np
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC
from src.networks.networks import ActorNetwork, CriticNetwork

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.core.core import Core
from src.algorithm.deep_actor_critic.boosted_rl import BRL,BRLReset
from mushroom_rl.core import Logger, Core
# from mushroom_rl.environments.dm_control_env import DMControl
from src.envs.dm_control_env import DMControl
from mushroom_rl.utils.dataset import compute_J, parse_dataset
import wandb
from mushroom_rl.core import Agent
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC


def experiment(alg):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)


    # MDP
    horizon = 1000
    gamma = 0.99
    mdp = DMControl('hopper', 'hop', horizon, gamma, use_pixels=False)

    # # agent = alg.load("checkpoint/walker/transfer_1_Q_10000_steps_policy")
    # initial_replay_size = 3 * horizon
    # max_replay_size = int(1e6)
    # batch_size = 256
    # n_features = 256
    # warmup_transitions = 10 * horizon
    # tau = 0.005
    # lr_alpha = 3e-4
    #
    # use_cuda = torch.cuda.is_available()
    #
    # # Approximator
    # actor_input_shape = mdp.info.observation_space.shape
    # actor_mu_params = dict(network=ActorNetwork,
    #                        n_features=n_features,
    #                        input_shape=actor_input_shape,
    #                        output_shape=mdp.info.action_space.shape,
    #                        use_cuda=use_cuda)
    # actor_sigma_params = dict(network=ActorNetwork,
    #                           n_features=n_features,
    #                           input_shape=actor_input_shape,
    #                           output_shape=mdp.info.action_space.shape,
    #                           use_cuda=use_cuda)
    #
    # actor_optimizer = {'class': optim.Adam,
    #                    'params': {'lr': 1e-4}}
    #
    # critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    # critic_params = dict(network=CriticNetwork,
    #                      optimizer={'class': optim.Adam,
    #                                 'params': {'lr': 3e-4}},
    #                      loss=F.mse_loss,
    #                      n_features=n_features,
    #                      input_shape=critic_input_shape,
    #                      output_shape=(1,),
    #                      use_cuda=use_cuda)
    #
    # # Agent
    # agent = SAC(mdp.info, actor_mu_params, actor_sigma_params,
    #             actor_optimizer, critic_params, batch_size, initial_replay_size,
    #             max_replay_size, warmup_transitions, tau, lr_alpha,
    #             log_std_min=-20, log_std_max=2, critic_fit_params=None)


    # Agent
    agent = Agent.load("checkpoint/hopper_hop/sac_nomimal_exp_0_epoch_1990_0")
    # Algorithm
    core = Core(agent, mdp)
    # # core.learn(n_steps=1, n_steps_per_fit=1)
    # agent.save("checkpoint/walker/sac_nomimal_exp_test", full_save=True)
    # agent =alg.load("checkpoint/walker/sac_nomimal_exp_test")
    # agent.save("checkpoint/walker/sac_nomimal_exp_test", full_save=True)
    # agent = alg.load("checkpoint/walker/sac_nomimal_exp_test")


    # input()
    dataset =core.evaluate(n_episodes=3, render=True)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    logger.epoch_info(0, J=J, R=R, entropy=E)


if __name__ == '__main__':

    for _ in range(1):
        experiment(alg=SAC)