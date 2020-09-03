import gym
import torch
import json
import os
import yaml
import shutil
from tqdm import trange
from mime.metalearners import MAMLTRPO, MAMLVIME, E_MAMLTRPO
from mime.baseline import LinearFeatureBaseline
from mime.samplers import MultiTaskSampler
from mime.utils.logger import log_returns, log_trajectories
from mime.utils.helpers import get_policy_for_env, get_input_size, get_dynamics_for_env, get_eta
from mime.utils.reinforcement_learning import get_returns

import random
import numpy as np

import dowel
from dowel import logger, tabular

import matplotlib
matplotlib.use('Agg')

def main(args):
    if args.use_vime and args.use_inv_vime:
        raise NotImplementedError("Choose one between VIME and (inverse) VIME")

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if os.path.exists(args.output_folder):
            shutil.rmtree(args.output_folder)
        os.makedirs(args.output_folder)

        policy_filename = os.path.join(args.output_folder, 'policy.th')
        dynamics_filename = os.path.join(args.output_folder, 'dynamics.th')
        config_filename = os.path.join(args.output_folder, 'config.json')

        text_log_file = os.path.join(args.output_folder, 'train_log.txt')
        tabular_log_file = os.path.join(args.output_folder, 'train_result.csv')

        # Set up logger
        logger.add_output(dowel.StdOutput())
        logger.add_output(dowel.TextOutput(text_log_file))
        logger.add_output(dowel.CsvOutput(tabular_log_file))
        logger.add_output(dowel.TensorBoardOutput(args.output_folder, x_axis='TotalEnvSteps'))
        logger.log('Logging to {}'.format(args.output_folder))

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Metaworld
    if config['env-name'].startswith('Metaworld'):
        env_name = config['env-name'].replace("Metaworld-","")
        metaworld = __import__('metaworld')
        # Set random seed
        metaworld.np.random.seed(args.seed)
        class_ = getattr(metaworld, env_name)
        metaworld_benchmark = class_()
        for name, env_cls in metaworld_benchmark.train_classes.items():
            env = env_cls()
            env.close()
        benchmark = metaworld_benchmark
    # Other gym envs
    else:
        env_name = config['env-name']
        env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
        env.close()
        benchmark = None

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()

    # Dynamics (for exploration)
    dynamics = get_dynamics_for_env(env, args.use_vime, args.use_inv_vime, args.device , config, benchmark=benchmark)
    inverse_dynamics = args.use_inv_vime

    # Eta
    eta = get_eta(args, config)
    eta_lr = config["eta-lr"] if "eta-lr" in config else None

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    epochs_counter = mp.Value('i', 0)

    # Sampler
    act_prev_mean = mp.Manager().list()
    obs_prev_mean = mp.Manager().list()
    pre_exploration_epochs = config["pre-epochs"] if "pre-epochs" in config else 1
    normalize_spaces = config["normalize-spaces"] if "normalize-spaces" in config else True

    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               dynamics=dynamics,
                               inverse_dynamics=inverse_dynamics,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers,
                               epochs_counter=epochs_counter,
                               act_prev_mean=act_prev_mean,
                               obs_prev_mean=obs_prev_mean,
                               # rew_prev_mean=rew_prev_mean,
                               eta=eta,
                               benchmark=benchmark,
                               pre_epochs=pre_exploration_epochs,
                               normalize_spaces=normalize_spaces,
                               add_noise=args.add_noise)

    use_dynamics = args.use_vime or args.use_inv_vime
    # Metalearner algo
    if use_dynamics and args.e_maml:
        raise NotImplementedError
    elif use_dynamics:
        metalearner = MAMLVIME(policy,
                               dynamics,
                               eta,
                               baseline,
                               fast_lr=config['fast-lr'],
                               dynamics_fast_lr=config['dyn-fast-lr'],
                               first_order=config['first-order'],
                               device=args.device,
                               epochs_counter=epochs_counter,
                               inverse_dynamics=inverse_dynamics,
                               gae_lambda=config['gae-lambda'],
                               eta_lr=eta_lr,
                               pre_epochs=pre_exploration_epochs,
                               benchmark=benchmark)
    elif args.e_maml:
        metalearner = E_MAMLTRPO(policy,
                               fast_lr=config['fast-lr'],
                               first_order=config['first-order'],
                               device=args.device)
    else:
        metalearner = MAMLTRPO(policy,
                               fast_lr=config['fast-lr'],
                               first_order=config['first-order'],
                               device=args.device)

    num_iterations = 0
    for batch in trange(config['num-batches'] + pre_exploration_epochs):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)

        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)

        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))

        tabular.record("TotalEnvSteps", logs["num_iterations"])
        tabular.record("Iteration", batch)

        log_returns(train_episodes, valid_episodes, batch,
                        log_dynamics=use_dynamics,
                        pre_exploration_epochs=pre_exploration_epochs,
                        benchmark=benchmark,
                        env=env,
                        env_name=env_name)
        log_trajectories(config['env-name'], args.output_folder, train_episodes, valid_episodes, batch)

        logger.log(tabular)

        logger.dump_all()

        with epochs_counter.get_lock():
            epochs_counter.value += 1

        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)

        # Save Dynamics
        if args.use_vime or args.use_inv_vime:
            if args.output_folder is not None:
                with open(dynamics_filename, 'wb') as f:
                    torch.save(dynamics.state_dict(), f)

        # Save data to continue training
        if ("obs_mean" not in config) and (epochs_counter.value == pre_exploration_epochs) and normalize_spaces:
            config["act_mean"] = act_prev_mean[-1]["mean"].tolist()
            config["act_std"] = act_prev_mean[-1]["std"].tolist()
            config["obs_mean"] = obs_prev_mean[-1]["mean"].tolist()
            config["obs_std"] = obs_prev_mean[-1]["std"].tolist()

        # Save eta
        if args.adapt_eta:
            config["adapted-eta"] = float(torch.sigmoid(eta.value.data))

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    logger.remove_all()


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None,
        help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')
    misc.add_argument('--use-vime', action='store_true', default=False,
                      help="Whether to use VIME (Variational Information Maximizing Exploration) or not (default: False)"
                      )
    misc.add_argument('--use-inv-vime', action='store_true', default=False,
                      help="Whether to use Inverse-Dynamics VIME (Variational Information Maximizing Exploration) or not (default: False)"
                      )
    misc.add_argument('--adapt-eta', action='store_true', default=False,
                      help="Whether to auto-adapt eta and balance exploration automatically (default: False)"
                      )
    misc.add_argument('--e-maml', action='store_true', default=False,
                      help="Whether to auto-adapt eta and balance exploration automatically (default: False)"
                      )
    misc.add_argument('--add-noise', action='store_true', default=False,
                      help="Whether to add a zero mean unit variance noise to the rewards (default: False)"
                      )


    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
