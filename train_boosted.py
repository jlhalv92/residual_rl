import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from src.algorithm.deep_actor_critic.boosted_rl import BRL,BRLReset
from mushroom_rl.core import Logger, Core
from src.envs.dm_control_env import DMControl
from mushroom_rl.utils.dataset import compute_J, parse_dataset
import wandb
from mushroom_rl.core import Agent
from src.networks.networks import Q
from src.networks.networks import ActorNetwork, CriticNetwork

from tqdm import trange


def experiment(alg, n_epochs, n_steps, n_steps_test, experiment_id):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)

    # MDP
    horizon = 500
    gamma = 0.99
    mdp = DMControl('walker', 'walk', horizon, gamma, use_pixels=False)
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="sac_walker_walk_mass_heavy",
    #     name="sac_walker_nomimal"
    # )
    # Settings
    initial_replay_size = 10000
    max_replay_size = int(1e6)
    batch_size = 256
    n_features = 400
    warmup_transitions = 15000
    tau = 0.005
    lr_alpha = 3e-4
    boosting = False
    # reset = True
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
    critic_params = dict(network=CriticNetwork,
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
                log_std_min=-3, log_std_max=2, use_entropy=False, target_entropy=None,
                 gauss_noise_cov=0.05, critic_fit_params=None, update_freq=1)

    if boosting:
        old_agent = alg.load("src/nominal_models/walker/nominal_walker")

        agent.setup_boosting(prior_agents=[old_agent],
                             use_kl_on_pi=False)

    # Algorithm
    core = Core(agent, mdp)

    # RUN
    agent._update_freq = 1
    core.learn(n_steps=2*initial_replay_size, n_steps_per_fit=1)
    # agent.policy.reset()

    for n in trange(n_epochs, leave=False):

        core.learn(n_steps=n_steps, n_steps_per_fit=1)

        dataset = core.evaluate(n_steps=n_steps_test, render=False)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        logger.epoch_info(n, J=J, R=R, entropy=E, Q=np.mean(agent._Q), old_Q=np.mean(agent._old_Q), rho=np.mean(agent._rho))
        logs_dict = {"RETURN": J, "REWARD": R, "Q":np.mean(agent._Q), "old_Q":np.mean(agent._old_Q), "rho":np.mean(agent._rho)}
        # wandb.log(logs_dict, step=n)
        agent._Q = []
        agent._old_Q = []
        agent._rho = []


        # if reset:
        #     if n%5 == 0 and n>0:
        #         core.agent.policy.reset()
        #         dataset = core.evaluate(n_steps=n_steps_test, render=False)
        #         s, *_ = parse_dataset(dataset)
        #         J = np.mean(compute_J(dataset, mdp.info.gamma))
        #         R = np.mean(compute_J(dataset))
        #         E = agent.policy.entropy(s)
        #
        #         logger.epoch_info(-1, J=J, R=R, entropy=E)


    # logger.info('Press a button to visualize pendulum')
    # input()
    # core.evaluate(n_episodes=5, render=True)
    agent.save("sac_walker_walker_residual_{}".format(experiment_id))
    wandb.finish()


if __name__ == '__main__':

    for experiment_id in range(5):
        experiment(alg=BRLReset, n_epochs=50, n_steps=10000, n_steps_test=2500, experiment_id=experiment_id)