import torch

from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence

from mime.samplers import MultiTaskSampler
from mime.metalearners.base import GradientBasedMetaLearner
from mime.utils.torch_utils import (weighted_mean, detach_distribution,
                                    to_numpy, vector_to_parameters)
from mime.utils.optimization import conjugate_gradient
from mime.utils.helpers import get_inputs_targets_dynamics
from mime.utils.reinforcement_learning import reinforce_loss
from dowel import tabular
import numpy as np



class MAMLVIME(GradientBasedMetaLearner):
    """Model-Agnostic Meta-Learning (MAML, [1]) for Reinforcement Learning
    application, with an outer-loop optimization based on TRPO [2].

    Parameters
    ----------
    policy : `mime.policies.Policy` instance
        The policy network to be optimized. Note that the policy network is an
        instance of `torch.nn.Module` that takes observations as input and
        returns a distribution (typically `Normal` or `Categorical`).

    fast_lr : float
        Step-size for the inner loop update/fast adaptation.

    num_steps : int
        Number of gradient steps for the fast adaptation. Currently setting
        `num_steps > 1` does not resample different trajectories after each
        gradient steps, and uses the trajectories sampled from the initial
        policy (before adaptation) to compute the loss at each step.

    first_order : bool
        If `True`, then the first order approximation of MAML is applied.

    device : str ("cpu" or "cuda")
        Name of the device for the optimization.

    References
    ----------
    .. [1] Finn, C., Abbeel, P., and Levine, S. (2017). Model-Agnostic
           Meta-Learning for Fast Adaptation of Deep Networks. International
           Conference on Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)

    .. [2] Schulman, J., Levine, S., Moritz, P., Jordan, M. I., and Abbeel, P.
           (2015). Trust Region Policy Optimization. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self,
                 policy,
                 dynamics,
                 eta,
                 baseline,
                 fast_lr=0.5,
                 first_order=False,
                 device='cpu',
                 epochs_counter=None,
                 inverse_dynamics=False,
                 gae_lambda=1.0,
                 dynamics_fast_lr=0.05,
                 eta_lr=100,
                 pre_epochs=None,
                 benchmark=None
                 ):
        super(MAMLVIME, self).__init__(policy, device=device)
        self.fast_lr = fast_lr
        self.first_order = first_order
        self.dynamics = dynamics
        self.baseline = baseline
        self.eta = eta
        self.epochs_counter = epochs_counter
        self.inverse_dynamics = inverse_dynamics
        self.dynamics_fast_lr = dynamics_fast_lr
        self.gae_lambda = gae_lambda
        self.eta_lr = eta_lr
        self.pre_epochs = pre_epochs
        self.benchmark=benchmark

    async def adapt(self, train_futures, first_order=None):
        if first_order is None:
            first_order = self.first_order
        # Loop over the number of steps of adaptation
        params = None
        for futures in train_futures:
            episodes = await futures
            if self.epochs_counter.value > self.pre_epochs and self.eta.requires_grad:
                episodes._rewards = (1.05 - self.eta.to_sigmoid()) * episodes._extrinsic_rewards + ( (self.eta.to_sigmoid() + 0.05) * episodes._intrinsic_rewards)
                # episodes._rewards = episodes._extrinsic_rewards + (self.eta.to_sigmoid() * episodes._intrinsic_rewards)
                self.baseline.fit(episodes)
                episodes.compute_advantages(self.baseline,
                                            gae_lambda=self.gae_lambda,
                                            normalize=True)
            inner_loss = reinforce_loss(self.policy,
                                        episodes,
                                        params=params)
            params = self.policy.update_params(inner_loss,
                                               params=params,
                                               step_size=self.fast_lr,
                                               first_order=first_order)
        return params

    async def adapt_dynamics(self, train_futures, first_order=None):
        if first_order is None:
            first_order = self.first_order

        device = self.device
        dyn_params = None
        # Loop over the number of steps of adaptation
        for futures in train_futures:
            episodes = await futures # TODO can remove?
            obs = episodes.observations
            act = episodes.actions
            obs_nxt = obs[1:]

            _inputs, _targets = get_inputs_targets_dynamics(obs, act, device, inverse=self.inverse_dynamics, benchmark=self.benchmark)

            # steps = _inputs.shape[0]
            # start = 0
            # end = 1
            # elbo_total = 0
            # for _ in range(steps):
            #     elbo = self.dynamics.loss_with_params(_inputs[start:end], _targets[start:end], params=dyn_params)
            #     elbo_total += elbo / _inputs.shape[0]
            #     start += 1
            #     end += 1

            elbo = 0
            for i in range(0, _inputs.shape[0], 128):
                elbo += self.dynamics.loss_with_params(_inputs[i:i + 128],
                                                      _targets[i:i + 128],
                                                      params=dyn_params) / _inputs.shape[0]
            dyn_params = self.dynamics.update_params(elbo,
                                                     params=dyn_params,
                                                     step_size=self.dynamics.learning_rate,
                                                     first_order=first_order)

            with torch.no_grad():
                _out = self.dynamics.pred_fn(_inputs)
                old_mse_loss = torch.mean((_out - _targets) ** 2)
                _out = self.dynamics.pred_sym_with_params(_inputs, dyn_params)
                new_mse_loss = torch.mean((_out - _targets) ** 2)
                # print("Out", _out)
                # print("Targets", _targets)
        return dyn_params, old_mse_loss, new_mse_loss

    def hessian_vector_product(self, kl, damping=1e-2):
        grads = torch.autograd.grad(kl,
                                    self.policy.parameters(),
                                    create_graph=True)
        flat_grad_kl = parameters_to_vector(grads)

        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v,
                                         self.policy.parameters(),
                                         retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    async def surrogate_loss(self, train_futures, valid_futures, old_pi=None, use_dynamics=False):
        first_order = (old_pi is not None) or self.first_order
        params = await self.adapt(train_futures,
                                  first_order=first_order)

        with torch.set_grad_enabled(old_pi is None):
            valid_episodes = await valid_futures

            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            log_ratio = (pi.log_prob(valid_episodes.actions)
                         - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)

            extrinsic_losses = None

            if self.epochs_counter.value > self.pre_epochs and self.eta.requires_grad:
                valid_episodes._rewards = valid_episodes._extrinsic_rewards
                # self.baseline.fit(valid_episodes)
                # valid_episodes.compute_advantages(self.baseline,
                #                                   gae_lambda=self.gae_lambda,
                #                                   normalize=True)
                extrinsic_losses = -weighted_mean(ratio * valid_episodes.rewards,
                                                  lengths=valid_episodes.lengths)
                extrinsic_losses = extrinsic_losses.mean()

                valid_episodes._rewards = (1.05 - self.eta.to_sigmoid()) * valid_episodes._extrinsic_rewards + (
                    (self.eta.to_sigmoid() + 0.05) * valid_episodes._intrinsic_rewards)
                # valid_episodes._rewards = valid_episodes._extrinsic_rewards + (
                #         self.eta.to_sigmoid() * valid_episodes._intrinsic_rewards)

                self.baseline.fit(valid_episodes)
                valid_episodes.compute_advantages(self.baseline,
                                                  gae_lambda=self.gae_lambda,
                                                  normalize=True)

            losses = -weighted_mean(ratio * valid_episodes.advantages,
                                    lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(pi, old_pi),
                                lengths=valid_episodes.lengths)

        if use_dynamics:
            # for k,v in self.dynamics.named_parameters():
            #     print("Before")
            #     print(k,v)
            #     break
            dyn_params, old_mse_loss, new_mse_loss = await self.adapt_dynamics(train_futures,
                                                   first_order=first_order)
            # for k,v in dyn_params.items():
            #     print("After")
            #     print(k,v)
            #     break

            device = self.device

            with torch.set_grad_enabled(True):
                obs = valid_episodes.observations
                act = valid_episodes.actions
                obs_nxt = obs[1:]

                _inputs, _targets = get_inputs_targets_dynamics(obs, act, device, inverse=self.inverse_dynamics, benchmark=self.benchmark)

                elbo = 0
                for i in range(0, _inputs.shape[0], 128):
                    elbo = self.dynamics.loss_with_params(_inputs[i:i + 128],
                                                          _targets[i:i + 128],
                                                          params=dyn_params) / _inputs.shape[0]
                # steps = _inputs.shape[0]
                # start = 0
                # end = 1
                # elbo_total = 0
                # for _ in range(steps):
                #     elbo = self.dynamics.loss_with_params(_inputs[start:end], _targets[start:end], params=dyn_params)
                #     elbo_total += elbo / _inputs.shape[0]
                #     start += 1
                #     end += 1

                return losses.mean(), kls.mean(), old_pi, elbo, old_mse_loss, new_mse_loss, extrinsic_losses
        else:
            return losses.mean(), kls.mean(), old_pi

    def step(self,
             train_futures,
             valid_futures,
             max_kl=1e-3,
             cg_iters=10,
             cg_damping=1e-2,
             ls_max_steps=10,
             ls_backtrack_ratio=0.5):
        num_tasks = len(train_futures[0])
        logs = {}

        # for k,v in self.dynamics.named_parameters():
        #     print("Before")
        #     print(k,v)
        #     break
        epochs_counter = self.epochs_counter.value

        if epochs_counter == 0: # self.pre_epochs:
            length = 50
        else:
            length = 1

        for _ in range(length):
            optim = torch.optim.Adam(params=self.dynamics.parameters(), lr=self.dynamics_fast_lr * num_tasks**2) # 0.05 # 0.01
            old_losses, old_kls, old_pis, dyn_losses, old_mse_losses, new_mse_losses, extrinsic_losses = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=None, use_dynamics=True)
                for (train, valid) in zip(zip(*train_futures), valid_futures)])
            dyn_loss = sum(dyn_losses) / num_tasks
            old_mse_loss = sum(old_mse_losses) / num_tasks
            new_mse_loss = sum(new_mse_losses) / num_tasks
            if _ == 0:
                tabular.record("Dynamics/OldMSELoss", np.float32(old_mse_loss))
            if _ == length -1:
                tabular.record("Dynamics/NewMSELoss", np.float32(new_mse_loss))
            dyn_loss.backward(retain_graph=True)
            optim.step()


        # for k,v in self.dynamics.named_parameters():
        #     print("After")
        #     print(k,v)
        #     break


        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)


        old_loss = sum(old_losses) / num_tasks

        if self.eta.requires_grad:
            if epochs_counter > self.pre_epochs:
            # if old_loss != 0:
                # Eta update
                extrinsic_loss = sum(extrinsic_losses) / num_tasks
                grads = torch.autograd.grad(extrinsic_loss,     # old_loss,
                                            self.eta.value,
                                            retain_graph=True)
                # print(grads)
                # self.eta = torch.sigmoid(torch.log(self.eta/(1-self.eta)) - self.fast_lr * grads[0])
                self.eta.value.data.copy_(self.eta.value.data - np.clip(grads[0] * self.eta_lr / torch.sigmoid(self.eta.value.data), -0.3, 0.3))  # * self.fast_lr
                print(grads[0])
            else:
                pass
            eta = float(np.float32(torch.sigmoid(self.eta.value.data)))
            tabular.record("Eta", eta)

        grads = torch.autograd.grad(old_loss,
                                    self.policy.parameters(),
                                    retain_graph=True)
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        old_kl = sum(old_kls) / num_tasks
        hessian_vector_product = self.hessian_vector_product(old_kl,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product,
                                     grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir,
                              hessian_vector_product(stepdir, retain_graph=False))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())

            losses, kls, _ = self._async_gather([
                self.surrogate_loss(train, valid, old_pi=old_pi)
                for (train, valid, old_pi)
                in zip(zip(*train_futures), valid_futures, old_pis)])

            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

        return logs
