import numpy as np

from gym.vector.utils import concatenate, create_empty_array

import numpy as np
from copy import deepcopy

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.vector.utils import concatenate, create_empty_array
import random

class SyncVectorEnv_(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.

    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.

    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.

    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def __init__(self,
                 # env_fns,
                 num_envs,
                 observation_space,
                 action_space,
                 copy=True):
        # self.env_fns = env_fns
        # self.envs = []
        self.copy = copy

        # if (observation_space is None) or (action_space is None):
        #     observation_space = observation_space or self.envs[0].observation_space
        #     action_space = action_space or self.envs[0].action_space
        super(SyncVectorEnv_, self).__init__(num_envs=num_envs,
                                            observation_space=observation_space, action_space=action_space)

        # self._check_observation_spaces()
        self.observations = create_empty_array(self.single_observation_space,
                                               n=self.num_envs, fn=np.zeros)
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def reset_wait(self):
        self._dones[:] = False
        observations = []
        for env in self.envs:
            observation = env.reset()
            observations.append(observation)
        concatenate(observations, self.observations, self.single_observation_space)

        return np.copy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            if self._dones[i]:
                observation = env.reset()
            observations.append(observation)
            infos.append(info)
        concatenate(observations, self.observations, self.single_observation_space)

        return (deepcopy(self.observations) if self.copy else self.observations,
                np.copy(self._rewards), np.copy(self._dones), infos)

    def close_extras(self, **kwargs):
        if hasattr(self, "envs"):
            [env.close() for env in self.envs]

    def _check_observation_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                # Adding new lines here
                if type(env.observation_space) == type(self.single_observation_space) and \
                        env.observation_space.shape == self.single_observation_space.shape:
                    continue
                else:
                    break
        else:
            return True
        raise RuntimeError('Some environments have an observation space '
                           'different from `{0}`. In order to batch observations, the '
                           'observation spaces from all environments must be '
                           'equal.'.format(self.single_observation_space))

class MetaworldSyncVectorEnv(SyncVectorEnv_):
    def __init__(self,
                 metaworld_benchmark,
                 num_envs,
                 observation_space=None,
                 action_space=None,
                 **kwargs):
        self.metaworld_benchmark = metaworld_benchmark
        self.num_envs = num_envs
        # self.env_name_list = []
        #
        # env_fns = []
        # for name, env_cls in metaworld_benchmark.train_classes.items():
        #     env_fns.append(env_cls)
        #     self.env_name_list.append(name)
        super(MetaworldSyncVectorEnv, self).__init__(num_envs,
                                            observation_space,
                                            action_space,
                                            **kwargs)
        # for env in self.envs:
        #     if not hasattr(env.unwrapped, 'reset_task'):
        #         raise ValueError('The environment provided is not a '
        #                          'meta-learning environment. It does not have '
        #                          'the method `reset_task` implemented.')

    @property
    def dones(self):
        return self._dones

    def reset_task(self, task):
        # training_envs = []
        if hasattr(self, "envs"):
        	del self.envs

        self.env_name = task.env_name

        if task.env_name in self.metaworld_benchmark.train_classes:
            env_cls = self.metaworld_benchmark.train_classes[task.env_name]
        elif task.env_name in self.metaworld_benchmark.test_classes:
            env_cls = self.metaworld_benchmark.test_classes[task.env_name]
        else:
            raise ValueError
        self.envs = [ env_cls() for _ in range(self.num_envs)]
        for env in self.envs:
            env.set_task(task)
        # for name, env_cls in ml10.train_classes.items():
        #     env = env_cls()
        #     task = random.choice([task for task in ml10.train_tasks
        #                           if task.env_name == name])
        #     env.set_task(task)
        #     training_envs.append(env)
        #
        # for name, env in zip(self.env_name_list, self.envs):
        #     task = random.choice([task for task in self.metaworld_benchmark.train_tasks
        #                           if task.env_name == name])
        #     env.set_task(task)
        #
        # for env in self.envs:
        #     env.unwrapped.reset_task(task)

    def step_wait(self):
        observations_list, infos = [], []
        batch_ids, j = [], 0
        num_actions = len(self._actions)
        rewards = np.zeros((num_actions,), dtype=np.float_)
        for i, env in enumerate(self.envs):
            if self._dones[i]:
                continue

            action = self._actions[j]
            observation, rewards[j], self._dones[i], info = env.step(action)
            batch_ids.append(i)

            infos.append(info)
            if not self._dones[i]:
                observations_list.append(observation)
            j += 1
        assert num_actions == j

        if observations_list:
            observations = create_empty_array(self.single_observation_space,
                                              n=len(observations_list),
                                              fn=np.zeros)
            concatenate(observations_list,
                        observations,
                        self.single_observation_space)
        else:
            observations = None

        return (observations, rewards, np.copy(self._dones),
                {'batch_ids': batch_ids, 'infos': infos, 'env_name': self.env_name})

