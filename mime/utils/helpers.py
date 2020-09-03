import gym
import torch

from functools import reduce
from operator import mul

from mime.dynamics import BNN
from mime.policies import CategoricalMLPPolicy, NormalMLPPolicy


class EtaParameter(torch.nn.Module):
    def __init__(self, value, adapt_eta=False):
        super(EtaParameter, self).__init__()
        if adapt_eta:
            self.value = torch.nn.Parameter(value)
        else:
            self.value = value
        self.requires_grad = adapt_eta

    def to_sigmoid(self):
        return torch.sigmoid(self.value)

def get_policy_for_env(env, hidden_sizes=(100, 100), nonlinearity='relu'):
    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    input_size = get_input_size(env)
    nonlinearity = getattr(torch, nonlinearity)

    if continuous_actions:
        output_size = reduce(mul, env.action_space.shape, 1)
        policy = NormalMLPPolicy(input_size,
                                 output_size,
                                 hidden_sizes=tuple(hidden_sizes),
                                 nonlinearity=nonlinearity)
    else:
        output_size = env.action_space.n
        policy = CategoricalMLPPolicy(input_size,
                                      output_size,
                                      hidden_sizes=tuple(hidden_sizes),
                                      nonlinearity=nonlinearity)
    return policy

def get_dynamics_for_env(env, use_vime, use_inv_vime, device, config, benchmark=None):
    action_dim = get_action_dim(env)
    n_batches = 1  # config['num-steps'] * config['fast-batch-size']

    if "dyn-lr" not in config:
        config["dyn-lr"] = config["dyn-fast-lr"]

    # Dynamics (VIME)
    if use_vime:
        n_in = env.observation_space.shape[0] + action_dim if not benchmark else env.observation_space.shape[0] + action_dim - 6
        n_out = env.observation_space.shape[0] if not benchmark else env.observation_space.shape[0] - 6
        dynamics = BNN(
            n_in=n_in,
            n_hidden=config["dyn-hidden"],
            n_out=n_out,
            prior_sd=config["dyn-prior-sd"],
            n_batches=n_batches,
            learning_rate=config["dyn-lr"],  # Half-Cheetah 0.0001 # 2DPoint 0.00001
            second_order_update=config["dyn-second-order"],
            trans_func=torch.nn.Tanh()
        ).to(device)
        dynamics.share_memory()
    # Inverse Dynamics
    elif use_inv_vime:
        raise NotImplementedError
        # dynamics = BNN(
        #     n_in=env.observation_space.shape[0] * 2,
        #     n_hidden=config["dyn-hidden"],
        #     n_out=action_dim,
        #     n_batches=n_batches,
        #     learning_rate=config["dyn-lr"],  # Half-Cheetah 0.0001 # 2DPoint 0.00001 #2dPoint_alt 0.0001
        #     second_order_update=config["dyn-second-order"], # TODO: act_func
        # ).to(device)
        # dynamics.share_memory()
    else:
        dynamics = None
    return dynamics

def get_eta(args, config):
    if args.adapt_eta:
        eta_value = torch.Tensor([config['adaptable-eta']])   #001])
        eta_value = torch.log(eta_value/(1 - eta_value))
        eta = EtaParameter(eta_value, adapt_eta=True)
        eta.share_memory()
    else:
        if 'eta' in config:
            eta_value = torch.Tensor([config['adaptable-eta']])  # 001])
            eta_value = torch.log(eta_value / (1 - eta_value))
            eta = EtaParameter(eta_value, adapt_eta=False)
            eta.share_memory()
        else:
            eta = None
    return eta

def get_input_size(env):
    return reduce(mul, env.observation_space.shape, 1)

def get_action_dim(env):
    if env.action_space.__class__.__name__ == "Discrete":
        action_dim = env.action_space.n
    elif env.action_space.__class__.__name__ == "Box":
        action_dim = env.action_space.shape[0]
    elif env.__class__.__name__ == "MultiBinary":
        action_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError
    return action_dim

def get_inputs_targets_dynamics(obs, act, device, inverse=False, benchmark=None):
    if benchmark:
        obs = obs[:,:,:-6]
    obs_nxt = obs[1:]
    if inverse:
        _inputs = torch.cat([obs[:-1], obs_nxt], dim=2).view(-1, obs.shape[2] * 2).to(device)
        _targets = act[:-1].view(-1, act.shape[2]).to(device)
    else:
        _inputs = torch.cat([obs[:-1], act[:-1]], dim=2).view(-1, obs.shape[2] + act.shape[2]).to(device)
        _targets = obs_nxt.view(-1, obs.shape[2]).to(device)
    return _inputs, _targets
