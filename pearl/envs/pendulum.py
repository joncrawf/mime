import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import random

from . import register_env

def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

@register_env('pendulum')
class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, sparse=False, dt=0.1, m=1., l=1., randomize_tasks=False, n_tasks=-1):
        self.g = 10.
        self.max_speed = 20.
        self.max_torque = 2.
        self.dt = dt
        self.m = m
        self.l = l
        self.viewer = None
        self.sparse = sparse

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

        num_train_tasks = 20
        params_list = []  # self.np_random.uniform(self.low, self.high, size=(num_tasks, 2))
        training_env_params = [
            (0.5, 0.4),
            (0.5, 0.5),
            (0.5, 0.7),
            (0.7, 0.4),
            (0.7, 0.5),
            (0.7, 0.7)
        ]
        for _ in range(num_train_tasks):
            # env = env_cls()
            index = self.np_random.randint(low=0, high=len(training_env_params)-1)
            params = training_env_params[index]
            # env.set_task(task)
            # training_envs.append(env)
            params_list.append(params)
        test_env_params = [
            (0.6, 0.6),
            (0.6, 0.8),
            (0.8, 0.6),
            (0.8, 0.8)
        ]
        for _ in range(len(test_env_params)):
            # env = env_cls()
            # params = random.choice(test_env_params)
            # env.set_task(task)
            # training_envs.append(env)
            params_list.append(params)

        self.tasks = [{'params': params} for params in params_list]

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        params_list = []  # self.np_random.uniform(self.low, self.high, size=(num_tasks, 2))
        training_env_params = [
            (0.5, 0.4),
            (0.5, 0.5),
            (0.5, 0.7),
            (0.7, 0.4),
            (0.7, 0.5),
            (0.7, 0.7)
        ]
        for _ in range(num_tasks):
            # env = env_cls()
            index = self.np_random.randint(low=0, high=len(training_env_params)-1)
            params = training_env_params[index]
            # env.set_task(task)
            # training_envs.append(env)
            params_list.append(params)

        tasks = [{'params': params} for params in params_list]
        return tasks

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx

        self.m = self._task['params'][0]
        self.l = self._task['params'][1]
        self.reset()

    def reset(self):
        # high = np.array([np.pi, 1])
        # self.state = self.np_random.uniform(low=-high, high=high)
        init_angle = self.np_random.uniform(low=3., high=3.28)
        init_vel = self.np_random.uniform(low=-0.01, high=0.01)
        self.state = np.float32([init_angle, init_vel])
        self.last_u = None
        return self._get_obs()

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering


        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        #     return (((x + np.pi) % (2 * np.pi)) - np.pi)
        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        cos_theta = np.cos(newth)
        if self.sparse:
            success = np.sqrt(2 * (self.l ** 2) * (1.0 - cos_theta)) < 0.0005 * self.l  # 0.0001
            reward = float(success)
            info = {"task": self._task, "success": success}
        else:
            # reward = - costs
            success = np.sqrt(2 * (self.l ** 2) * (1.0 - cos_theta)) < 0.0005 * self.l
            reward = -np.sqrt(2 * (self.l ** 2) * (1.0 - np.cos(newth)))
            info = {"task": self._task, "success": success}

            # info = {"task": self._task}

        # reward = -np.sqrt(2 * (self.l ** 2) * (1.0 - np.cos(newth)))

        # success = np.sqrt(2 * (self.l ** 2) * (1.0 - np.cos(newth))) < 0.04

        return self._get_obs(), reward, False, info     #   -costs as rewards

    # def step(self, u):
    #     th, thdot = self.state  # th := theta
    #
    #     g = self.g
    #     m = self.m
    #     l = self.l
    #     dt = self.dt
    #
    #     u = self.max_torque * u
    #     u = np.clip(u, -self.max_torque, self.max_torque)[0]
    #     self.last_u = u  # for rendering
    #     costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    #
    #     newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
    #     newth = th + newthdot * dt
    #     newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)  # pylint: disable=E1111
    #
    #     done = np.abs(newthdot) > self.max_speed
    #
    #     self.state = np.array([newth, newthdot])
    #
    #     reward = -np.sqrt(2 * (self.l ** 2) * (1.0 - np.cos(newth)))
    #
    #     success = np.sqrt(2 * (self.l ** 2) * (1.0 - np.cos(newth))) < 0.08
    #
    #     info = {"success": success, "task": self._task}
    #
    #     return self._get_obs(), reward, done, info

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None

    def check_if_solved(self, states):
        cos_theta = np.cos(states[:, 0])
        dist = np.sqrt(2 * (self.l ** 2) * (1.0 - cos_theta)) # < 0.08
        solved_list = list((dist[-10:] < 0.00005).numpy())
        solved = all(solved_list)
        return solved

    def sample_test_tasks(self, num_tasks):
        params_list = []  # self.np_random.uniform(self.low, self.high, size=(num_tasks, 2))
        test_env_params = [
            (0.6, 0.6),
            (0.6, 0.8),
            (0.8, 0.6),
            (0.8, 0.8)
        ]
        for _ in range(len(test_env_params)):
            # env = env_cls()
            params = random.choice(test_env_params)
            # env.set_task(task)
            # training_envs.append(env)
            params_list.append(params)

        tasks = [{'params': params} for params in params_list]
        return tasks

@register_env('pendulum-sparse')
class PendulumSparseEnv(PendulumEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, sparse=False, dt=0.1, m=1., l=1., randomize_tasks=False, n_tasks=-1):
        super(PendulumSparseEnv, self).__init__(sparse=True)