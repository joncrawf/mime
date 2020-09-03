import torch
import torch.multiprocessing as mp
import asyncio
import threading
import time
import random

from datetime import datetime, timezone
from copy import deepcopy

from mime.samplers.sampler import Sampler, make_env
from mime.envs.utils.mw_sync_env import MetaworldSyncVectorEnv
from mime.envs.utils.sync_vector_env import SyncVectorEnv
from mime.episode import BatchEpisodes
from mime.utils.reinforcement_learning import reinforce_loss
from mime.utils.helpers import get_inputs_targets_dynamics
from mime.utils.curiosity import compute_dynamics_curiosity, compute_inv_dynamics_curiosity
import numpy as np


def _create_consumer(queue, futures, loop=None):
    if loop is None:
        loop = asyncio.get_event_loop()
    while True:
        data = queue.get()
        if data is None:
            break
        index, step, episodes = data
        future = futures if (step is None) else futures[step]
        if not future[index].cancelled():
            loop.call_soon_threadsafe(future[index].set_result, episodes)


class MultiTaskSampler(Sampler):
    """Vectorized sampler to sample trajectories from multiple environements.

    Parameters
    ----------
    env_name : str
        Name of the environment. This environment should be an environment
        registered through `gym`. See `maml.envs`.

    env_kwargs : dict
        Additional keywork arguments to be added when creating the environment.

    batch_size : int
        Number of trajectories to sample from each task (ie. `fast_batch_size`).

    policy : `mime.policies.Policy` instance
        The policy network for sampling. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    baseline : `mime.baseline.LinearFeatureBaseline` instance
        The baseline. This baseline is an instance of `nn.Module`, with an
        additional `fit` method to fit the parameters of the model.

    env : `gym.Env` instance (optional)
        An instance of the environment given by `env_name`. This is used to
        sample tasks from. If not provided, an instance is created from `env_name`.

    seed : int (optional)
        Random seed for the different environments. Note that each task and each
        environement inside every process use different random seed derived from
        this value if provided.

    num_workers : int
        Number of processes to launch. Note that the number of processes does
        not have to be equal to the number of tasks in a batch (ie. `meta_batch_size`),
        and can scale with the amount of CPUs available instead.
    """

    def __init__(self,
                 env_name,
                 env_kwargs,
                 batch_size,
                 policy,
                 baseline,
                 dynamics=None,
                 inverse_dynamics=False,
                 env=None,
                 seed=None,
                 num_workers=1,
                 epochs_counter=None,
                 act_prev_mean=None,
                 obs_prev_mean=None,
                 eta=None,
                 benchmark=None,
                 pre_epochs=-1,
                 normalize_spaces=True,
                 add_noise=False
                 ):
        super(MultiTaskSampler, self).__init__(env_name,
                                               env_kwargs,
                                               batch_size,
                                               policy,
                                               seed=seed,
                                               env=env)
        # Metaworld
        self.benchmark = benchmark

        ### Dynamics
        self.env_name = env_name
        self.epochs_counter = epochs_counter
        self.pre_epochs = pre_epochs

        self.dynamics = dynamics
        if self.dynamics is not None:
            self.kl_previous = mp.Manager().list()
            kl_previous_lock = mp.Manager().RLock()
            dynamics_lock = mp.Lock()
        else:
            dynamics_lock = None
            self.kl_previous = None
            kl_previous_lock = None

        self.inverse_dynamics = inverse_dynamics

        self.act_prev_mean = act_prev_mean
        self.obs_prev_mean = obs_prev_mean

        act_prev_lock = mp.Manager().RLock()
        obs_prev_lock = mp.Manager().RLock()

        self.num_workers = num_workers

        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        policy_lock = mp.Lock()

        self.workers = [SamplerWorker(index,
                                      env_name,
                                      env_kwargs,
                                      batch_size,
                                      self.env.observation_space,
                                      self.env.action_space,
                                      self.policy,
                                      deepcopy(baseline),
                                      self.seed,
                                      self.task_queue,
                                      self.train_episodes_queue,
                                      self.valid_episodes_queue,
                                      policy_lock,
                                      # Queues and Epochs
                                      epochs_counter=epochs_counter,
                                      pre_epochs=pre_epochs,
                                      act_prev_lock=act_prev_lock,
                                      obs_prev_lock=obs_prev_lock,
                                      act_prev_mean=self.act_prev_mean,
                                      obs_prev_mean=self.obs_prev_mean,
                                      # Dynamics
                                      dynamics=self.dynamics,
                                      dynamics_lock=dynamics_lock,
                                      kl_previous=self.kl_previous,
                                      kl_previous_lock=kl_previous_lock,
                                      inverse_dynamics=self.inverse_dynamics,
                                      eta=eta,
                                      # Metaworld
                                      benchmark=benchmark,
                                      normalize_spaces=normalize_spaces,
                                      add_noise=add_noise
                                      )
                        for index in range(num_workers)]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def sample_tasks(self, num_tasks):
        if self.benchmark is not None:
            training_tasks = []
            for _ in range(num_tasks):
                task = random.choice(self.benchmark.train_tasks)
                training_tasks.append(task)
            return training_tasks
        else:
            return self.env.unwrapped.sample_tasks(num_tasks)

    def sample_test_tasks(self, num_tasks):
        if self.benchmark is not None:
            testing_tasks = []

            env_names_list = list(self.benchmark.test_classes.keys())
            while len(env_names_list) > 0:
                task = random.choice(self.benchmark.test_tasks)
                env_name = task.env_name
                if env_name in env_names_list:
                    testing_tasks.append(task)
                    env_names_list.remove(env_name)
            return testing_tasks
        else:
            return self.env.unwrapped.sample_test_tasks(num_tasks)

    def sample_async(self, tasks, **kwargs):
        if self._waiting_sample:
            raise RuntimeError('Calling `sample_async` while waiting '
                               'for a pending call to `sample_async` '
                               'to complete. Please call `sample_wait` '
                               'before calling `sample_async` again.')

        for index, task in enumerate(tasks):
            self.task_queue.put((index, task, kwargs))

        num_steps = kwargs.get('num_steps', 1)
        futures = self._start_consumer_threads(tasks,
                                               num_steps=num_steps)
        self._waiting_sample = True
        return futures

    def sample_wait(self, episodes_futures):
        if not self._waiting_sample:
            raise RuntimeError('Calling `sample_wait` without any '
                               'prior call to `sample_async`.')

        async def _wait(train_futures, valid_futures):
            # Gather the train and valid episodes
            train_episodes = await asyncio.gather(*[asyncio.gather(*futures)
                                                    for futures in train_futures])
            valid_episodes = await asyncio.gather(*valid_futures)
            return (train_episodes, valid_episodes)

        samples = self._event_loop.run_until_complete(_wait(*episodes_futures))
        self._join_consumer_threads()
        self._waiting_sample = False
        return samples

    def sample(self, tasks, **kwargs):
        futures = self.sample_async(tasks, **kwargs)
        return self.sample_wait(futures)

    @property
    def train_consumer_thread(self):
        if self._train_consumer_thread is None:
            raise ValueError()
        return self._train_consumer_thread

    @property
    def valid_consumer_thread(self):
        if self._valid_consumer_thread is None:
            raise ValueError()
        return self._valid_consumer_thread

    def _start_consumer_threads(self, tasks, num_steps=1):
        # Start train episodes consumer thread
        train_episodes_futures = [[self._event_loop.create_future() for _ in tasks]
                                  for _ in range(num_steps)]
        self._train_consumer_thread = threading.Thread(target=_create_consumer,
                                                       args=(self.train_episodes_queue, train_episodes_futures),
                                                       kwargs={'loop': self._event_loop},
                                                       name='train-consumer')
        self._train_consumer_thread.daemon = True
        self._train_consumer_thread.start()

        # Start valid episodes consumer thread
        valid_episodes_futures = [self._event_loop.create_future() for _ in tasks]
        self._valid_consumer_thread = threading.Thread(target=_create_consumer,
                                                       args=(self.valid_episodes_queue, valid_episodes_futures),
                                                       kwargs={'loop': self._event_loop},
                                                       name='valid-consumer')
        self._valid_consumer_thread.daemon = True
        self._valid_consumer_thread.start()

        return (train_episodes_futures, valid_episodes_futures)

    def _join_consumer_threads(self):
        if self._train_consumer_thread is not None:
            self.train_episodes_queue.put(None)
            self.train_consumer_thread.join()

        if self._valid_consumer_thread is not None:
            self.valid_episodes_queue.put(None)
            self.valid_consumer_thread.join()

        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def close(self):
        if self.closed:
            return

        for _ in range(self.num_workers):
            self.task_queue.put(None)
        self.task_queue.join()
        self._join_consumer_threads()

        self.closed = True


class SamplerWorker(mp.Process):
    def __init__(self,
                 index,
                 env_name,
                 env_kwargs,
                 batch_size,
                 observation_space,
                 action_space,
                 policy,
                 baseline,
                 seed,
                 task_queue,
                 train_queue,
                 valid_queue,
                 policy_lock,
                 dynamics=None,
                 dynamics_lock=None,
                 epochs_counter=None,
                 kl_previous=None,
                 kl_previous_lock=None,
                 inverse_dynamics=False,
                 act_prev_mean=None,
                 obs_prev_mean=None,
                 act_prev_lock=None,
                 obs_prev_lock=None,
                 eta=None,
                 benchmark=None,
                 pre_epochs=None,
                 normalize_spaces=True,
                 add_noise=False
                 ):
        super(SamplerWorker, self).__init__()

        if benchmark is not None:
            self.envs = MetaworldSyncVectorEnv(benchmark,
                                               batch_size,
                                              observation_space=observation_space,
                                              action_space=action_space)
        else:
            env_fns = [make_env(env_name, env_kwargs=env_kwargs)
                       for _ in range(batch_size)]
            self.envs = SyncVectorEnv(env_fns,
                                      observation_space=observation_space,
                                      action_space=action_space)
            self.envs.seed(None if (seed is None) else seed + index * batch_size)

        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline

        self.task_queue = task_queue
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.policy_lock = policy_lock

        self.epochs_counter = epochs_counter
        self.pre_epochs = pre_epochs

        self.dynamics = dynamics
        self.dynamics_lock = dynamics_lock
        self.kl_previous = kl_previous
        self.kl_previous_lock = kl_previous_lock

        self.inverse_dynamics = inverse_dynamics

        self.act_prev_mean = act_prev_mean
        self.obs_prev_mean = obs_prev_mean
        self.act_prev_lock = act_prev_lock
        self.obs_prev_lock = obs_prev_lock

        self.eta = eta
        self.benchmark = benchmark
        self.normalize_spaces = normalize_spaces

        self.add_noise = add_noise

    def sample(self,
               index,
               num_steps=1,
               fast_lr=0.5,
               gamma=0.95,
               gae_lambda=1.0,
               device='cpu'):
        # Sample the training trajectories with the initial policy and adapt the
        # policy to the task, based on the REINFORCE loss computed on the
        # training trajectories. The gradient update in the fast adaptation uses
        # `first_order=True` no matter if the second order version of MAML is
        # applied since this is only used for sampling trajectories, and not
        # for optimization.
        params = None
        dyn_params = None
        inv_dyn_params = None
        for step in range(num_steps):
            train_episodes = self.create_episodes(params=params,
                                                  gamma=gamma,
                                                  gae_lambda=gae_lambda,
                                                  device=device)
            train_episodes.log('_enqueueAt', datetime.now(timezone.utc))
            # QKFIX: Deep copy the episodes before sending them to their
            # respective queues, to avoid a race condition. This issue would
            # cause the policy pi = policy(observations) to be miscomputed for
            # some timesteps, which in turns makes the loss explode.
            self.train_queue.put((index, step, deepcopy(train_episodes)))

            with self.policy_lock:
                loss = reinforce_loss(self.policy, train_episodes, params=params)
                params = self.policy.update_params(loss,
                                                   params=params,
                                                   step_size=fast_lr,
                                                   first_order=True)

            if self.dynamics is not None:
                with self.dynamics_lock:
                    obs = train_episodes.observations
                    act = train_episodes.actions

                    if act.dim() < 3:
                        act = act.view(act.shape[0], act.shape[1], 1)


                    _inputs, _targets = get_inputs_targets_dynamics(obs, act, device, inverse=self.inverse_dynamics, benchmark=self.benchmark)

                    # Trick to prevent exploding loss
                    elbo = 0
                    for i in range(0, _inputs.shape[0], 128):
                        elbo += self.dynamics.loss_with_params(_inputs[i:i+128],
                                                               _targets[i:i+128],
                                                               params=dyn_params) / _inputs.shape[0]
                    dyn_params = self.dynamics.update_params(elbo,
                                                             params=dyn_params,
                                                             step_size=self.dynamics.learning_rate,
                                                             first_order=True)

        # Sample the validation trajectories with the adapted policy
        valid_episodes = self.create_episodes(params=params,
                                              gamma=gamma,
                                              gae_lambda=gae_lambda,
                                              device=device,
                                              dyn_params=dyn_params,
                                              inv_dyn_params=inv_dyn_params)
        valid_episodes.log('_enqueueAt', datetime.now(timezone.utc))
        self.valid_queue.put((index, None, deepcopy(valid_episodes)))

    def create_episodes(self,
                        params=None,
                        gamma=0.95,
                        gae_lambda=1.0,
                        device='cpu',
                        dyn_params=None,
                        inv_dyn_params=None):
        episodes = BatchEpisodes(batch_size=self.batch_size,
                                 gamma=gamma,
                                 device=device)
        episodes.log('_createdAt', datetime.now(timezone.utc))
        episodes.log('process_name', self.name)

        with self.epochs_counter.get_lock():
            epochs_counter = self.epochs_counter.value

        t0 = time.time()

        info_collected = False
        success = None
        for item in self.sample_trajectories(params=params):
            observations, actions, rewards, batch_ids, infos = item
            if info_collected is False and len(infos["infos"]) > 0:
                episodes.log("infos", infos["infos"])
                info_collected = True
                if "env_name" in infos:
                    episodes.log("env_name", infos["env_name"])
            for info in infos["infos"]:
                if "success" in info and (not success):
                    success = info["success"] == True

            episodes.append(observations, actions, rewards, batch_ids)

        if success is not None:
            episodes.log("success", success)

        episodes.log('observations', episodes.observations)
        episodes.log('return', episodes.rewards.sum(dim=0).mean())

        if self.normalize_spaces:
            if epochs_counter <= self.pre_epochs:
                with self.act_prev_lock:
                    if epochs_counter < self.pre_epochs: # TODO: parameter?
                        self.act_prev_mean.append({"mean": episodes.actions.mean(dim=[0, 1]), "std": np.clip(episodes.actions.std(dim=[0, 1]), 1e-6, np.inf)})
                    mean = []
                    std = []
                    for i in range(len(self.act_prev_mean)):
                        mean.append(self.act_prev_mean[i]["mean"].view(1, -1))
                        std.append(self.act_prev_mean[i]["std"].view(1, -1))
                    mean_vec = torch.Tensor(len(self.act_prev_mean), *self.act_prev_mean[0]["mean"].shape)
                    std_vec = torch.Tensor(len(self.act_prev_mean), *self.act_prev_mean[0]["std"].shape)
                    torch.cat(mean, dim=0, out=mean_vec)
                    torch.cat(std, dim=0, out=std_vec)
                    act_mean = mean_vec.mean(dim=0)
                    act_std = std_vec.mean(dim=0)
                    if epochs_counter == self.pre_epochs:
                        self.act_prev_mean.append({"mean": act_mean, "std": act_std})
                    if len(self.act_prev_mean) == 1:
                        act_std = torch.Tensor([1.])
                    for index, value in enumerate(act_std):
                        if value < 1e-2:
                            act_std[index] = 1.
                    episodes._actions = (episodes.actions - act_mean) / (act_std)

                with self.obs_prev_lock:
                    if epochs_counter < self.pre_epochs:
                        self.obs_prev_mean.append({"mean": episodes.observations.mean(dim=[0, 1]) , "std": np.clip(episodes.observations.std(dim=[0, 1]), 1e-6, np.inf)})
                    mean = []
                    std = []
                    for i in range(len(self.obs_prev_mean)):
                        mean.append(self.obs_prev_mean[i]["mean"].view(1, -1))
                        std.append(self.obs_prev_mean[i]["std"].view(1, -1))
                    mean_vec = torch.Tensor(len(self.obs_prev_mean), *self.obs_prev_mean[0]["mean"].shape)
                    std_vec = torch.Tensor(len(self.obs_prev_mean), *self.obs_prev_mean[0]["std"].shape)
                    torch.cat(mean, dim=0, out=mean_vec)
                    torch.cat(std, dim=0, out=std_vec)
                    obs_mean = mean_vec.mean(dim=0)
                    obs_std = std_vec.mean(dim=0)
                    if epochs_counter == self.pre_epochs:
                        self.obs_prev_mean.append({"mean": obs_mean, "std": obs_std})
                    if len(self.obs_prev_mean) == 1:
                        obs_std = torch.Tensor([1.])
                    if len(self.obs_prev_mean) > (self.batch_size * 10):
                        self.obs_prev_mean.pop(0)
                    episodes._observations = (episodes.observations - obs_mean) / (obs_std)
            else:
                with self.act_prev_lock:
                    act_mean = self.act_prev_mean[-1]["mean"]
                    act_std = self.act_prev_mean[-1]["std"]
                with self.obs_prev_lock:
                    obs_mean = self.obs_prev_mean[-1]["mean"]
                    obs_std = self.obs_prev_mean[-1]["std"]
                episodes._actions = (episodes.actions - act_mean) / act_std
                episodes._observations = (episodes.observations - obs_mean) / obs_std


        episodes._rewards = (episodes.rewards - episodes._rewards.mean()) / (episodes._rewards.std() + 1e-8)
        if self.add_noise:
            episodes._rewards += 0.1 * torch.from_numpy(np.random.normal(loc=0., scale=1., size=episodes._rewards.shape))


        kl = torch.Tensor([0.])

        if (self.dynamics is not None) and (epochs_counter > self.pre_epochs):  # and (dyn_params is None):
            with self.dynamics_lock:
                dynamics = deepcopy(self.dynamics)
                dynamics.replace_params(params=dyn_params)
                dynamics.opt = torch.optim.Adam(params=dynamics.parameters(), lr=dynamics.learning_rate)
            kl = compute_dynamics_curiosity(episodes, dynamics, device, dyn_params, inverse=self.inverse_dynamics, benchmark=self.benchmark)
            episodes.log('intrinsic_return', kl.sum(dim=0).mean())
            kl = (kl - kl.mean()) / (kl.std() + 1e-8)


        if (self.dynamics is not None) and (epochs_counter > self.pre_epochs):
            # Add KL as intrinsic reward to external reward
            eta = self.eta.value.data

            episodes._intrinsic_rewards = kl                
            episodes._extrinsic_rewards = episodes.rewards


            episodes._rewards = (1.05 - torch.sigmoid(eta)) * episodes.rewards + ( (torch.sigmoid(eta) + 0.05) * episodes._intrinsic_rewards)

            episodes.log('combined_return', episodes.rewards.sum(dim=0).mean())

            # eta *= eta_discount

        episodes.log('duration', time.time() - t0)

        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline,
                                    gae_lambda=gae_lambda,
                                    normalize=True)
        return episodes

    def sample_trajectories(self, params=None):
        observations = self.envs.reset()
        with torch.no_grad():
            while not self.envs.dones.all():
                observations_tensor = torch.from_numpy(observations)
                pi = self.policy(observations_tensor, params=params)
                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()

                new_observations, rewards, dones, infos = self.envs.step(actions)

                batch_ids = infos['batch_ids']
                yield (observations, actions, rewards, batch_ids, infos)
                observations = new_observations

    def run(self):
        while True:
            data = self.task_queue.get()

            if data is None:
                self.envs.close()
                self.task_queue.task_done()
                break

            index, task, kwargs = data
            self.envs.reset_task(task)
            self.sample(index, **kwargs)
            self.task_queue.task_done()
