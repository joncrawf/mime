import numpy as np
import torch

def compute_dynamics_curiosity(episodes, dynamics, device, dyn_params, inverse=False, benchmark=None):
    if inverse:
        raise NotImplementedError
        return compute_inv_dynamics_curiosity(episodes, dynamics, device)
    second_order_update = dynamics.second_order_update
    kl_batch_size = 1
    n_itr_update = 1

    # Iterate over all paths and compute intrinsic reward by updating the
    # model on each observation, calculating the KL divergence of the new
    # params to the old ones, and undoing this operation.
    obs = episodes.observations if not benchmark else episodes.observations[:,:,:-6]  # observation should be already normalized
    act = episodes.actions
    rew = episodes.rewards

    num_trajectories = obs.shape[1]

    kl = torch.zeros(rew.shape)
    for t in range(num_trajectories):
        # inputs = (o,a), target = o'

        # Replace for each trajectory
        dynamics.replace_params(params=dyn_params)
        dynamics.opt = torch.optim.Adam(params=dynamics.parameters(), lr=dynamics.learning_rate) #/ episodes.observations.shape[0])

        obs_nxt = obs[1:, t]
        _inputs = torch.cat([obs[:-1, t], act[:-1, t]], dim=1)
        _targets = obs_nxt

        _inputs = _inputs.to(device)
        _targets = _targets.to(device)
        # KL vector assumes same shape as reward.
        for k in range(int(np.ceil(obs.shape[0] / float(kl_batch_size)))):

            dynamics.save_old_params()
            start = k * kl_batch_size
            end = np.minimum(
                (k + 1) * kl_batch_size, obs.shape[0] - 1)

            if second_order_update:
                # We do a line search over the best step sizes using
                # step_size * invH * grad
                #                 best_loss_value = np.inf
                for step_size in [0.01]:
                    loss_value = dynamics.train_update_fn(
                        _inputs[start:end], _targets[start:end], second_order_update, step_size)
                    loss_value = loss_value.detach()
                    kl_div = np.clip(loss_value, 0, 1000)
                    # If using replay pool, undo updates.
                    # if use_replay_pool:
                    #    dynamics.reset_to_old_params()
            else:
                # Update model weights based on current minibatch.
                for _ in range(n_itr_update):
                    dynamics.train_update_fn(
                        _inputs[start:end], _targets[start:end], second_order_update)
                # Calculate current minibatch KL.
                kl_div = np.clip(
                    float(dynamics.f_kl_div_closed_form().detach()), 0, 1000)

            for k in range(start, end):
                kl[k][t] = kl_div

            # If using replay pool, undo updates.
            dynamics.reset_to_old_params()

    # Last element in KL vector needs to be replaced by second last one
    # because the actual last observation has no next observation.
    kl[-1] = kl[-2]
    return kl


def compute_inv_dynamics_curiosity(episodes, inv_dynamics, device):
    second_order_update = True
    kl_batch_size = 1
    use_replay_pool = True
    n_itr_update = 1

    # Iterate over all paths and compute intrinsic reward by updating the
    # model on each observation, calculating the KL divergence of the new
    # params to the old ones, and undoing this operation.
    obs = episodes.observations  # observation should be already normalized
    act = episodes.actions
    rew = episodes.rewards

    num_trajectories = obs.shape[1]

    kl = torch.zeros(rew.shape)
    for t in range(num_trajectories):
        # inputs = (o,a), target = o'
        obs_nxt = obs[1:, t]
        _inputs = torch.cat([obs[:-1, t], obs_nxt], dim=1)
        _targets = act[:-1, t]

        _inputs = _inputs.to(device)
        _targets = _targets.to(device)
        # KL vector assumes same shape as reward.
        for k in range(int(np.ceil(obs.shape[0] / float(kl_batch_size)))):

            start = k * kl_batch_size
            end = np.minimum(
                (k + 1) * kl_batch_size, obs.shape[0] - 1)

            if second_order_update:
                # We do a line search over the best step sizes using
                # step_size * invH * grad
                #                 best_loss_value = np.inf
                for step_size in [0.01]:
                    loss_value = inv_dynamics.train_update_fn(
                        _inputs[start:end], _targets[start:end], second_order_update, step_size)
                    loss_value = loss_value.detach()
                    kl_div = np.clip(loss_value, 0, 1000)
                    # If using replay pool, undo updates.
                    if use_replay_pool:
                        inv_dynamics.reset_to_old_params()
            else:
                # Update model weights based on current minibatch.
                for _ in range(n_itr_update):
                    inv_dynamics.train_update_fn(
                        _inputs[start:end], _targets[start:end], second_order_update)
                # Calculate current minibatch KL.
                kl_div = np.clip(
                    float(inv_dynamics.f_kl_div_closed_form().detach()), 0, 1000)

            for k in range(start, end):
                kl[k][t] = kl_div

            # If using replay pool, undo updates.
            if use_replay_pool:
                inv_dynamics.reset_to_old_params()

    # Last element in KL vector needs to be replaced by second last one
    # because the actual last observation has no next observation.
    kl[0] = kl[1]
    kl[-2] = kl[-1]
    return kl
