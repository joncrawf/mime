import numpy as np
import gym

from gym import spaces
from gym.utils import seeding


class Navigation2DEnv(gym.Env):
    """2D navigation problems, as described in [1]. The code is adapted from 
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py

    At each time step, the 2D agent takes an action (its velocity, clipped in
    [-0.1, 0.1]), and receives a penalty equal to its L2 distance to the goal 
    position (ie. the reward is `-distance`). The 2D navigation tasks are 
    generated by sampling goal positions from the uniform distribution 
    on [-0.5, 0.5]^2.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, task={}, low=-0.5, high=0.5):
        super(Navigation2DEnv, self).__init__()
        self.low = low
        self.high = high

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._task = task
        self._goal = task.get('goal', np.zeros(2, dtype=np.float32))
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        goals = self.np_random.uniform(self.low, self.high, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        assert self.action_space.contains(action)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward = -np.sqrt(x ** 2 + y ** 2)
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {'task': self._task}

class Navigation2DEnvSparse(gym.Env):
    """2D navigation problems, as described in [1]. The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/maml_examples/point_env_randgoal.py

    At each time step, the 2D agent takes an action (its velocity, clipped in
    [-0.1, 0.1]), and receives a rewards which is the inverse of
    its L2 distance to the goal when it is close to the goal position.
    (ie. the reward is `1/distance`). The 2D navigation tasks are
    generated by sampling goal positions from the uniform distribution
    on [-0.5, 0.5]^2.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, task={}, low=-0.5, high=0.5, sparse=True):
        super(Navigation2DEnvSparse, self).__init__()
        self.low = low
        self.high = high

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.1, high=0.1,
            shape=(2,), dtype=np.float32)

        self._task = task
        self.sparse = sparse
        self._goal = task.get('goal', np.zeros(2, dtype=np.float32))
        self._state = np.zeros(2, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        goals = np.empty((num_tasks,2))
        i = 0
        while i < num_tasks:
            goal = self.np_random.randn(1, 2) * self.high
            # if (self.low < np.abs(goal[0, 0]) < self.high or self.low < np.abs(goal[0, 1]) < self.high) \
            #         and -self.high < goal[0, 0] < self.high and -self.high < goal[0, 1] < self.high:
            distance = np.sqrt( goal[0,0] ** 2 + goal[0,1] ** 2)
            if self.low < distance < self.high:
                goals[i] = goal
                i += 1
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def sample_test_tasks(self, num_tasks):
        goals = []
        dog = False
        if dog:
            diag = 8.0 * np.sqrt(2) / 2
            goals.append(np.array([0,8.0]))
            goals.append(np.array([8.0,0]))
            goals.append(np.array([diag,diag]))
            goals.append(np.array([0,8.0]))
            goals.append(np.array([8.0,0]))
            goals.append(np.array([diag,diag]))
            goals.append(np.array([0,8.0]))
            goals.append(np.array([8.0,0]))
            goals.append(np.array([diag,diag]))
        elif self.low == 5.0 and self.high == 5.5:
            medium = (self.high + self.low)/2
            diag_medium = medium * np.sqrt(2) / 2
            goals.append(np.array([0,medium]))
            goals.append(np.array([medium,0]))
            goals.append(np.array([0,-medium]))
            goals.append(np.array([-medium,0]))
            goals.append(np.array([diag_medium,diag_medium]))
            goals.append(np.array([diag_medium,-diag_medium]))
            goals.append(np.array([-diag_medium,-diag_medium]))
            goals.append(np.array([-diag_medium,diag_medium]))
        elif self.low == 8.0 and self.high == 9.0:
            medium = (self.high + self.low) / 2
            diag_medium = medium * np.sqrt(2) / 2
            goals.append(np.array([0,medium]))
            goals.append(np.array([medium,0]))
            goals.append(np.array([0,-medium]))
            goals.append(np.array([-medium,0]))
            goals.append(np.array([diag_medium,diag_medium]))
            goals.append(np.array([diag_medium,-diag_medium]))
            goals.append(np.array([-diag_medium,-diag_medium]))
            goals.append(np.array([-diag_medium,diag_medium]))
        elif self.low == 5.0 and self.high == 9.0:
            medium = 5.5
            diag_medium = medium * np.sqrt(2) / 2
            goals.append(np.array([0,medium]))
            goals.append(np.array([medium,0]))
            goals.append(np.array([0,-medium]))
            goals.append(np.array([-medium,0]))
            goals.append(np.array([diag_medium,diag_medium]))
            goals.append(np.array([diag_medium,-diag_medium]))
            goals.append(np.array([-diag_medium,-diag_medium]))
            goals.append(np.array([-diag_medium,diag_medium]))
            medium = 8.5
            diag_medium = medium * np.sqrt(2) / 2
            goals.append(np.array([0,medium]))
            goals.append(np.array([medium,0]))
            goals.append(np.array([0,-medium]))
            goals.append(np.array([-medium,0]))
            goals.append(np.array([diag_medium,diag_medium]))
            goals.append(np.array([diag_medium,-diag_medium]))
            goals.append(np.array([-diag_medium,-diag_medium]))
            goals.append(np.array([-diag_medium,diag_medium]))
        else:
            goals = np.empty((num_tasks, 2))
            i = 0
            while i < num_tasks:
                goal = self.np_random.randn(1, 2) * self.high
                # if (self.low < np.abs(goal[0, 0]) < self.high or self.low < np.abs(goal[0, 1]) < self.high) \
                #         and -self.high < goal[0, 0] < self.high and -self.high < goal[0, 1] < self.high:
                distance = np.sqrt(goal[0, 0] ** 2 + goal[0, 1] ** 2)
                if self.low < distance < self.high:
                    goals[i] = goal
                    i += 1
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']

    def reset(self, env=True):
        self._state = np.zeros(2, dtype=np.float32)
        return self._state

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)

        assert self.action_space.contains(action)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        distance = np.sqrt(x ** 2 + y ** 2)


        if self.sparse:
            if distance < 1: #(np.abs(x) < 1.) and (np.abs(y) < 1.):
                reward = +1.    # / (distance + 1e-8)
                success = True
            else:
                success = False
                reward = + 0.
            info = {'task': self._task, 'success': float(success)}
        else:
            reward = -distance
            if distance < 1: #(np.abs(x) < 1.) and (np.abs(y) < 1.):
                success = True
            else:
                success = False
            info = {'task': self._task, 'success': float(success)}

        done = False    # ((np.abs(x) < 0.05) and (np.abs(y) < 0.05))
        has_end = False

        if has_end and success:
            done = True

        return self._state, reward, done, info