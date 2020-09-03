from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='mime.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='mime.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'AntVel-v2',
    entry_point='mime.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'mime.envs.mujoco.ant:AntVelEnv'}
)

register(
    'AntDir-v2',
    entry_point='mime.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'mime.envs.mujoco.ant:AntDirEnv'}
)

register(
    'AntPos-v1',
    entry_point='mime.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'mime.envs.mujoco.ant:AntPosEnv'}
)

register(
    'HalfCheetahVel-v2',
    entry_point='mime.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'mime.envs.mujoco.half_cheetah:HalfCheetahVelEnv'}
)

register(
    'HalfCheetahDir-v2',
    entry_point='mime.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'mime.envs.mujoco.half_cheetah:HalfCheetahDirEnv'}
)

register(
    'HalfCheetahDirSparse-v2',
    entry_point='mime.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'mime.envs.mujoco.half_cheetah:HalfCheetahDirSparseEnv'}
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='mime.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

register(
    '2DNavigationSparse-v0',
    entry_point='mime.envs.navigation:Navigation2DEnvSparse',
    max_episode_steps=100
)

register(
    '2DNavigationSparseLong-v0',
    entry_point='mime.envs.navigation:Navigation2DEnvSparse',
    max_episode_steps=300
)


register(
    'DogNavigation2D-v0',
    entry_point='mime.envs.dog_navigation:DogNavigation2D',
    max_episode_steps=1000
)


# Pendulum
# ----------------------------------------

register(
    'MetaPendulum-v0',
    entry_point='mime.envs.pendulum:PendulumEnv',
    max_episode_steps=50
)
