import numpy as np
from mushroom_rl.core import Agent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from src.algorithm.deep_actor_critic.boosted_rl import BRL,BRLReset
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC
from mushroom_rl.core import Logger
from src.core.core import Core
from mushroom_rl.environments.dm_control_env import DMControl
from mushroom_rl.utils.dataset import compute_J, parse_dataset
import wandb
from src.networks.networks import ActorNetwork, CriticNetwork
from tqdm import trange
from omegaconf import DictConfig, OmegaConf
import hydra

def experiment(alg, experiment_id, cfg=None):

    if cfg is None:
        assert "ERROR, configuration file is missing"

    np.random.seed()
    n_epochs = cfg.num_epochs
    n_steps = cfg.num_steps
    n_episodes_test = cfg.num_eval_episodes

    env = cfg.env.robot
    task = cfg.env.task
    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__ + " ENV: {}_{}".format(env, task))

    # MDP
    horizon = cfg.env.horizon
    gamma = cfg.env.discount
    logging = cfg.log_wandb
    conf_dict = OmegaConf.to_container(cfg.agent, resolve=True)

    mdp = DMControl(env, task, horizon, gamma, use_pixels=False)
    if logging:
        wandb.init(
            # set the wandb project where this run will be logged
            project=cfg.wandb_project,
            name=cfg.wandb_exp_name,
            config=conf_dict
        )
    # Settings

    initial_replay_size = cfg.initial_replay_size
    max_replay_size = int(cfg.num_train_steps)
    batch_size = cfg.agent.model.params.batch_size
    n_features = cfg.agent.critic.params.n_features
    warmup_transitions = cfg.warmup_transitions
    tau = cfg.agent.critic.params.critic_tau
    lr_alpha = cfg.agent.model.params.alpha_lr
    freq_checkpoints = cfg.freq_checkpoints

    use_cuda = True if (torch.cuda.is_available() and cfg.cuda) else False

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
                       'params': {'lr': cfg.agent.actor.params.lr}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': cfg.agent.critic.params.lr}},
                         loss=F.mse_loss,
                         n_features=n_features,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)


    # Agent
    agent = alg(mdp.info, actor_mu_params, actor_sigma_params,
                actor_optimizer, critic_params, batch_size, initial_replay_size,
                max_replay_size, warmup_transitions, tau, lr_alpha,
                log_std_min=cfg.agent.actor.params.log_std_bounds[0],
                log_std_max=cfg.agent.actor.params.log_std_bounds[1],
                critic_fit_params=None)

    # agent =Agent.load("checkpoint/quadruped_walk/sac_nomimal_exp_0_epoch_1999_3")


    # Algorithm
    core = Core(agent, mdp)

    # RUN
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    for n in trange(n_epochs, leave=False):

        core.learn(n_steps=n_steps, n_steps_per_fit=1)

        dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
        s, *_ = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, mdp.info.gamma))
        R = np.mean(compute_J(dataset))
        E = agent.policy.entropy(s)

        logger.epoch_info(n, J=J, R=R, entropy=E)
        logs_dict = {"RETURN": J, "REWARD": R, "Entropy":E}
        if logging:
            wandb.log(logs_dict, step=n)

        if n%freq_checkpoints == 0 or n==n_epochs-1:
            agent.save("{}/{}_{}/sac_nomimal_exp_{}_epoch_{}".format(cfg.checkpoint_dir,
                                                                     env,
                                                                     task,
                                                                     experiment_id,
                                                                     n))

    if logging:
        wandb.finish()

@hydra.main(version_base=None, config_path='configs', config_name="train")
def run_exp(cfg: DictConfig):
    main(cfg)

def main(cfg: DictConfig = None):

    alg_map = {"sac": SAC,  # Mappings from strings to algorithms
                "BRL": None}

    alg = alg_map[cfg.agent.model.name]
    for experiment_id in range(cfg.seeds):
        experiment(alg=alg, experiment_id=experiment_id, cfg=cfg)


if __name__ == '__main__':
    run_exp()


