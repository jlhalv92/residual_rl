import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.algorithm.deep_actor_critic.resnet_residual_rl import ResnetResidualRL
from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.core import Core, Logger
from src.envs.dm_control_env import DMControl
from mushroom_rl.utils.dataset import compute_J, parse_dataset
from src.networks.networks import CriticNetwork, ActorNetwork, QRESLIM, QRESLIM2,QRESLIM3
from tqdm import trange
import wandb
from joblib import Parallel, delayed
import time


def experiment(alg, n_epochs, n_steps, n_episodes_test, run_id, target_speed, residuals, reference_epoch, curriculum_id, stop_log):
    np.random.seed()

    logger = Logger(alg.__name__, results_dir=None)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + alg.__name__)
    critics = [QRESLIM, QRESLIM2, QRESLIM3]
    # MDP
    horizon = 500
    gamma = 0.99
    mdp = DMControl('walker',
                    'run',
                    horizon,
                    gamma,
                    use_pixels=False)
    log = True
    mdp.env.task._move_speed = target_speed

    if log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="walker_run_comparison",
            name="residual_resnet_curriculum_3_steps_3_5_8_no_policy"
        )

    # Settings
    initial_replay_size = 0.01 * n_steps*n_epochs
    max_replay_size = n_steps*n_epochs
    batch_size = 400
    n_features = 400
    # warmup_transitions = 0.015*n_steps*n_epochs
    warmup_transitions = 0.012*n_steps*n_epochs

    tau = 0.005
    lr_alpha = 3e-4

    use_cuda = True

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
    critic_params = dict(network=critics[curriculum_id],
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
    boosting = True
    core = Core(agent, mdp)



    if boosting:
        old_agents = []
        for r_path in residuals:
            old_agents.append(alg.load(r_path))

        use_policy = False

        if curriculum_id > 0:
            use_policy = False

        agent.setup_residual(prior_agents=old_agents,
                             use_kl_on_pi=False,
                             kl_on_pi_alpha=0.08,
                             copy_weights=True,
                             use_policy=use_policy)



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
        Q = np.mean(agent._Q)
        rho = np.mean(agent._rho)
        old_q = np.mean(agent._old_Q)

        logger.epoch_info(n+1+reference_epoch, J=J, R=R, entropy=E, q=Q, rho=rho, old_q=old_q)
        agent._Q = []
        agent._rho = []
        agent._old_Q = []
        logs_dict = {"RETURN": J, "REWARD": R}

        if log:
            wandb.log(logs_dict, step=n+reference_epoch)

    # logger.info('Press a button to visualize pendulum')
    # input()
    # core.evaluate(n_episodes=5, render=True)
    agent.save("checkpoint/walker_rho_3_steps_no_policy_3_5_8_{}_{}".format(curriculum_id, run_id))
    if stop_log:
        wandb.finish()

if __name__ == '__main__':

    # logger = Logger("test", results_dir=None)
    # logger.strong_line()
    # for i in range(5):
    #     experiment(alg=ResnetResidualRL, n_epochs=50, n_steps=5000, n_episodes_test=10, run_id=i)

    for i in range(5):
        residuals = ["src/nominal_models/walker/nominal_walker"]
        target_speed_list = [3., 5., 8.]
        epochs = [15, 15, 20]
        ref_epochs = [0, 15, 30]
        rhos = ["checkpoint/walker_rho_3_steps_no_policy_3_5_8_{}_{}".format(j, i) for j in range(2)]
        residuals.extend(rhos)
        stop_log = False
        for j in range(len(target_speed_list)):
            if j == len(target_speed_list)-1:
                stop_log = True
            experiment(alg=ResnetResidualRL,
                       n_epochs=epochs[j],
                       n_steps=5000,
                       n_episodes_test=10,
                       run_id=i,
                       residuals=residuals[:j+1],
                       target_speed=target_speed_list[j],
                       reference_epoch=ref_epochs[j],
                       curriculum_id=j,
                       stop_log=stop_log)
    # Parallel(n_jobs=5)()
    # for x in range(5):
    #
    #     for y in range(6):
    # out = Parallel(n_jobs=-1)(delayed(experiment)(alg=ResnetResidualRL,
    #                                               n_epochs=50,
    #                                               n_steps=5000,
    #                                               n_episodes_test=10,
    #                                               run_id=experiment_id) for experiment_id in range(n_experiment))

