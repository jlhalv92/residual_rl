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
    mdp = DMControl('quadruped', 'walk', horizon, gamma, use_pixels=False)

    # Agent
    agent = Agent.load("checkpoint/quadruped_walk/sac_nomimal_exp_0_epoch_1940_0")
    # Algorithm
    core = Core(agent, mdp)

    dataset =core.evaluate(n_episodes=3, render=True)
    s, *_ = parse_dataset(dataset)

    J = np.mean(compute_J(dataset, mdp.info.gamma))
    R = np.mean(compute_J(dataset))
    E = agent.policy.entropy(s)
    logger.epoch_info(0, J=J, R=R, entropy=E)


if __name__ == '__main__':

    for _ in range(1):
        experiment(alg=SAC)