import mime.envs
import gym
import torch
import json
import numpy as np
from tqdm import trange
import shutil


from mime.baseline import LinearFeatureBaseline
from mime.samplers import MultiTaskSampler
from mime.utils.helpers import get_policy_for_env, get_input_size, get_dynamics_for_env, EtaParameter
from mime.utils.reinforcement_learning import get_returns

import random
import dowel
from dowel import logger, tabular

from mime.utils.logger import log_returns, log_trajectories
import matplotlib
matplotlib.use('Agg')


def main(args):

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            raise ValueError("The folder with the training files does not exist")

    policy_filename = os.path.join(args.output_folder, 'policy.th')
    dynamics_filename = os.path.join(args.output_folder, 'dynamics.th')
    config_filename = os.path.join(args.output_folder, 'config.json')
    # eval_filename = os.path.join(args.output_folder, 'eval.npz')

    text_log_file = os.path.join(args.output_folder, 'test_log.txt')
    tabular_log_file = os.path.join(args.output_folder, 'test_result.csv')

    output_test_folder =args.output_folder + "test" if args.output_folder[-1] == '/' else args.output_folder + "/test"

    if os.path.exists(output_test_folder):
        shutil.rmtree(output_test_folder)
    os.makedirs(output_test_folder)

    # Set up logger
    logger.add_output(dowel.StdOutput())
    logger.add_output(dowel.TextOutput(text_log_file))
    logger.add_output(dowel.CsvOutput(tabular_log_file))
    logger.add_output(dowel.TensorBoardOutput(output_test_folder, x_axis='Batch'))
    logger.log('Logging to {}'.format(output_test_folder))

    with open(config_filename, 'r') as f:
        config = json.load(f)

    seed = config["seed"] if "seed" in config else args.seed
    if seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)

    # Metaworld
    if config['env-name'].startswith('Metaworld'):
        env_name = config['env-name'].replace("Metaworld-","")
        metaworld = __import__('metaworld')
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

    with open(policy_filename, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    # Dynamics
    dynamics = get_dynamics_for_env(env, config['use_vime'], config['use_inv_vime'], args.device , config, benchmark=benchmark)
    inverse_dynamics = config['use_inv_vime']
    use_dynamics = config["use_vime"] or config["use_inv_vime"]

    if use_dynamics:
        with open(dynamics_filename, 'rb') as f:
            state_dict = torch.load(f, map_location=torch.device(args.device))
            dynamics.load_state_dict(state_dict)
        dynamics.share_memory()

    # Eta
    if config['adapt_eta']:
        eta_value = torch.Tensor([config["adapted-eta"]])
    else:
        eta_value = torch.Tensor([config["eta"]])
    eta_value = torch.log(eta_value / (1 - eta_value))
    eta = EtaParameter(eta_value, adapt_eta=config['adapt_eta'])
    eta.share_memory()


    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    normalize_spaces = config["normalize-spaces"] if "normalize-spaces" in config else True
    act_prev_mean = mp.Manager().list()
    obs_prev_mean = mp.Manager().list()

    # Sampler
    if normalize_spaces:
        obs_prev_mean.append({ "mean": torch.Tensor(config["obs_mean"]), "std" : torch.Tensor(config["obs_std"])})
        act_prev_mean.append({ "mean": torch.Tensor(config["act_mean"]), "std" : torch.Tensor(config["act_std"])})

    epochs_counter = mp.Value('i', 100)

    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'], # TODO
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
                               normalize_spaces=normalize_spaces)

    logs = {'tasks': []}
    train_returns, valid_returns = [], []
    for batch in trange(args.num_batches):
        tasks = sampler.sample_test_tasks(num_tasks=config['meta-batch-size'])
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=args.num_steps,
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        logs['tasks'].extend(tasks)
        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))

        logs['train_returns'] = np.concatenate(train_returns, axis=0)
        logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

        tabular.record("Batch", batch)

        log_returns(train_episodes, valid_episodes, batch,
                    log_dynamics=use_dynamics,
                    benchmark=benchmark,
                    env=env,
                    env_name=env_name,
                    is_testing=True)
        log_trajectories(config['env-name'], output_test_folder, train_episodes, valid_episodes, batch)

        logger.log(tabular)

        logger.dump_all()

        # with open(eval_filename + "_" + str(batch), 'wb') as f:
        #     np.savez(f, **logs)

    logger.remove_all()

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=3,
        help='number of batches (default: 3)')
    evaluation.add_argument('--num-steps', type=int, default=10,
        help='number of batches (default: 10)')
    # evaluation.add_argument('--meta-batch-size', type=int, default=40,
    #     help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str, required=True,
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count(),
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count()))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)
