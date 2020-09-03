import numpy as np
from dowel import tabular
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import matplotlib.image as mpimg


def plot_return_success(episodes, label):
    episodes_return = 0
    episodes_success = 0
    has_success = "success" in episodes[0]._logs
    # if success:
    #     num_trajectories = len(train_episodes[0][0]._logs["success"])
    for t in episodes:
        episodes_return += t._logs["return"]
        if has_success and t._logs["success"]:
            episodes_success += 1
    episodes_return = (episodes_return / len(episodes))
    tabular.record(f"{label}/Returns", np.float32(episodes_return))
    if has_success:
        tabular.record(f"{label}/Success", np.float32(episodes_success / len(episodes)))


def plot_dynamics_return(episodes, label):
    intrinsic_return = 0
    combined_return = 0

    for t in episodes:
        intrinsic_return += t._logs["intrinsic_return"]
        combined_return += t._logs["combined_return"]
    intrinsic_return = (intrinsic_return / len(episodes))
    combined_return = (combined_return / len(episodes))

    tabular.record(f"{label}/IntrinsicReturns", np.float32(intrinsic_return))
    tabular.record("Valid/CombinedReturns", np.float32(combined_return))

def plot_benchmark(benchmark, train_episodes, valid_episodes, is_testing):
    env_names_list = list(benchmark.train_classes.keys())
    if is_testing:
        env_names_list = list(benchmark.test_classes.keys())
    for train_step, train_eps in enumerate(train_episodes):
        envs = dict()
        for episodes in train_eps:
            env_name = episodes._logs["env_name"]
            if env_name in env_names_list:
                env_names_list.remove(env_name)
            if env_name not in envs:
                envs[env_name] = { "return" : np.float32(episodes._logs["return"]),
                                   "success" : float(episodes._logs["success"]),
                                   "count" : 1}
            else:
                envs[env_name]["return"] += np.float32(episodes._logs["return"])
                envs[env_name]["success"] += float(episodes._logs["success"])
                envs[env_name]["count"] += 1

        for env_name, env_stats in envs.items():
            tabular.record(f'Train-{train_step}/{env_name}/Returns', env_stats["return"] / env_stats["count"])
            tabular.record(f'Train-{train_step}/{env_name}/Success', env_stats["success"] / env_stats["count"])
        for env_name in env_names_list:
            tabular.record(f'Train-{train_step}/{env_name}/Returns', np.nan)
            tabular.record(f'Train-{train_step}/{env_name}/Success', np.nan)

    envs = dict()
    for episodes in valid_episodes:
        env_name = episodes._logs["env_name"]
        # env_names_list.remove(env_name)
        if env_name not in envs:
            envs[env_name] = {"return": np.float32(episodes._logs["return"]),
                              "success": float(episodes._logs["success"]),
                              "count": 1}
        else:
            envs[env_name]["return"] += np.float32(episodes._logs["return"])
            envs[env_name]["success"] += float(episodes._logs["success"])
            envs[env_name]["count"] += 1

    for env_name, env_stats in envs.items():
        tabular.record(f'Valid/{env_name}/Returns', env_stats["return"] / env_stats["count"])
        tabular.record(f'Valid/{env_name}/Success', env_stats["success"] / env_stats["count"])
    for env_name in env_names_list:
        tabular.record(f'Valid/{env_name}/Returns', np.nan)
        tabular.record(f'Valid/{env_name}/Success', np.nan)

def log_returns(train_episodes,
                valid_episodes,
                batch,
                log_dynamics=False,
                pre_exploration_epochs=-1,
                benchmark=None,
                is_testing=False,
                env=None,
                env_name=""):
    if env:
        if env_name == "MetaPendulum-v0":
            for train_step, train_eps in enumerate(train_episodes):
                if 'success' in train_eps[0]._logs:
                    break
                for train_episode in train_eps:
                    success = False
                    for traj_index in range(train_episode._logs['observations'].shape[2]):
                        if not success:
                            trajectory = train_episode._logs['observations'][:, :, traj_index]
                            success = env.check_if_solved(trajectory)
                    train_episode._logs['success'] = success

            for train_episode in valid_episodes:
                if 'success' in train_episode._logs:
                    break
                success = False
                for traj_index in range(train_episode._logs['observations'].shape[2]):
                    if not success:
                        trajectory = train_episode._logs['observations'][:, :, traj_index]
                        success = env.check_if_solved(trajectory)
                train_episode._logs['success'] = success

    for train_step, train_eps in enumerate(train_episodes):
        plot_return_success(train_eps, f'Train-{train_step}')

    plot_return_success(valid_episodes, "Valid")

    if log_dynamics:
        if batch > pre_exploration_epochs:
            for train_step, train_eps in enumerate(train_episodes):
                plot_dynamics_return(train_eps, f'Train-{train_step}')

            plot_dynamics_return(valid_episodes, "Valid")

        else:
            for train_step in range(len(train_episodes)):
                tabular.record(f"Train-{train_step}/IntrinsicReturns", 0)
                tabular.record(f"Train-{train_step}/CombinedReturns", 0)
            tabular.record("Valid/IntrinsicReturns", 0)
            tabular.record("Valid/CombinedReturns", 0)

    if benchmark:
        plot_benchmark(benchmark, train_episodes, valid_episodes, is_testing)

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def log_trajectories(env_name, output_folder, train_episodes, valid_episodes, batch, is_test=False):
    if env_name in ["2DNavigationSparse-v0","2DNavigationSparseLong-v0"]:
        if is_test:
            pass
        else:
            # Train
            figure, ax = plt.subplots(figsize=(8, 8))
            # ax.fill([-10, -10, 10, 10], [7.5, 10, 10, 7.5], color="red", alpha=0.3, lw=0)
            # ax.fill([-10, -10, 10, 10], [-7.5, -10, -10, -7.5], color="red", alpha=0.3, lw=0)
            # ax.fill([-7.5, -7.5, -10, -10], [-7.5, 7.5, 7.5, -7.5], color="red", alpha=0.3, lw=0)
            # ax.fill( [7.5, 7.5, 10, 10], [-7.5, 7.5, 7.5, -7.5], color="red", alpha=0.3, lw=0)

            num_tasks = len(train_episodes[0])
            colors = ['firebrick', 'chocolate', 'darkorange', 'gold', 'darkolivegreen', 'lightgreen', 'turquoise', 'teal', 'dodgerblue', 'purple']
            times_colors = int( num_tasks / len(colors)) + 1
            colors = (colors * times_colors)[:num_tasks]

            for i, color in enumerate(colors):
            # for train_episode in train_episodes[0]:
                train_episode = train_episodes[0][i]
                trajectories = train_episode._logs['observations'] # steps, n_trajectories, space_dims
                for t in range(trajectories.shape[1]):
                    plt.plot(trajectories[:, t, 0], trajectories[:, t, 1], color=color, alpha=0.4)
                for task_dict in train_episode._logs['infos']:
                    goal = task_dict["task"]["goal"]
                    circle_reward = plt.Circle(goal, 1., color=color, alpha=0.3)
                    # circle_goal = plt.Circle(goal, 0.05, color=color, alpha=0.6)
                    ax.add_artist(circle_reward)
                    # ax.add_artist(circle_goal)
                    break  # Because it is repeated several times

            circle_goals = plt.Circle([0,0], radius=7, fill=False, linewidth=2.0, linestyle='--', color='darkslateblue')
            ax.add_artist(circle_goals)

            plt.axis([-10, +10, -10, +10])

            image = plot_to_image(figure)

            logdir = output_folder + '/train_trajectories'
            file_writer = tf.summary.create_file_writer(logdir)

            with file_writer.as_default():
                tf.summary.image("TrainTrajectories", image, step=batch)

            # Valid
            figure, ax = plt.subplots(figsize=(8, 8))

            # ax.fill([-10, -10, 10, 10], [7.5, 10, 10, 7.5], color="red", alpha=0.3, lw=0)
            # ax.fill([-10, -10, 10, 10], [-7.5, -10, -10, -7.5], color="red", alpha=0.3, lw=0)
            # ax.fill([-7.5, -7.5, -10, -10], [-7.5, 7.5, 7.5, -7.5], color="red", alpha=0.3, lw=0)
            # ax.fill( [7.5, 7.5, 10, 10], [-7.5, 7.5, 7.5, -7.5], color="red", alpha=0.3, lw=0)

            for i, color in enumerate(colors):
                valid_episode = valid_episodes[i]
                # for valid_episode in valid_episodes:
                trajectories = valid_episode._logs['observations']  # steps, n_trajectories, space_dims
                for t in range(trajectories.shape[1]):
                    plt.plot(trajectories[:, t, 0], trajectories[:, t, 1], color=color, alpha=0.4)
                for task_dict in valid_episode._logs['infos']:
                    goal = task_dict["task"]["goal"]
                    circle_reward = plt.Circle(goal, 1., color=color, alpha=0.3)
                    # circle_goal = plt.Circle(goal, 0.05, color='blue', alpha=0.6)
                    ax.add_artist(circle_reward)
                    # ax.add_artist(circle_goal)
                    break

            circle_goals = plt.Circle([0,0], radius=7, fill=False, linewidth=2.0, linestyle='--', color='darkslateblue')
            ax.add_artist(circle_goals)

            plt.axis([-10, +10, -10, +10])

            image = plot_to_image(figure)

            with file_writer.as_default():
                tf.summary.image("ValidTrajectories", image, step=batch)
    if env_name == "DogNavigation2D-v0":
        # Train
        figure, ax = plt.subplots(figsize=(8, 8))

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        img = mpimg.imread('dog_house.png')
        gray = np.flipud(rgb2gray(img))

        plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1, origin='lower')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        # ax.fill([-10, -10, 10, 10], [7.5, 10, 10, 7.5], color="red", alpha=0.3, lw=0)
        # ax.fill([-10, -10, 10, 10], [-7.5, -10, -10, -7.5], color="red", alpha=0.3, lw=0)
        # ax.fill([-7.5, -7.5, -10, -10], [-7.5, 7.5, 7.5, -7.5], color="red", alpha=0.3, lw=0)
        # ax.fill( [7.5, 7.5, 10, 10], [-7.5, 7.5, 7.5, -7.5], color="red", alpha=0.3, lw=0)

        num_tasks = len(train_episodes[0])
        colors = ['darkorange', 'darkolivegreen', 'dodgerblue', 'purple']
        times_colors = int(num_tasks / len(colors)) + 1
        colors = (colors * times_colors)[:num_tasks]

        for i, color in enumerate(colors):
            # for train_episode in train_episodes[0]:
            train_episode = train_episodes[0][i]
            trajectories = train_episode._logs['observations']*10  # steps, n_trajectories, space_dims
            for t in range(trajectories.shape[1]):
                plt.plot(trajectories[:, t, 0], trajectories[:, t, 1], color=color, alpha=0.4)
            for task_dict in train_episode._logs['infos']:
                goal = task_dict["task"]["goal"]*10
                circle_reward = plt.Circle(goal, 5., color=color, alpha=0.3)
                # circle_goal = plt.Circle(goal, 0.05, color=color, alpha=0.6)
                ax.add_artist(circle_reward)
                # ax.add_artist(circle_goal)
                break  # Because it is repeated several times


        # # Create figure and axes
        # fig, ax = plt.subplots(1, figsize=(9, 9))
        #
        # # Display the image
        # ax.axes.get_xaxis().set_ticks([])
        # ax.axes.get_yaxis().set_ticks([])
        # ax.imshow(img, cmap=plt.get_cmap('gray'))
        # plt.plot([13, 14], [80] * 2)  # Start: 13x80
        # plt.plot([23, 24], [55] * 2)  # Goal1: 25x55
        # plt.plot([20, 21], [10] * 2)  # Goal2: 20x10
        # plt.plot([85, 86], [15] * 2)  # Goal3: 85x15
        # plt.plot([85, 86], [87] * 2)  # Goal4: 85x87
        # # plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1, )
        # # plt.show()
        # Create figure and axes

        image = plot_to_image(figure)

        logdir = output_folder + '/train_trajectories'
        file_writer = tf.summary.create_file_writer(logdir)

        with file_writer.as_default():
            tf.summary.image("TrainTrajectories", image, step=batch)

        # Valid
        figure, ax = plt.subplots(figsize=(8, 8))

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        img = mpimg.imread('dog_house.png')
        gray = np.flipud(rgb2gray(img))

        plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1, origin='lower')
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        # ax.fill([-10, -10, 10, 10], [7.5, 10, 10, 7.5], color="red", alpha=0.3, lw=0)
        # ax.fill([-10, -10, 10, 10], [-7.5, -10, -10, -7.5], color="red", alpha=0.3, lw=0)
        # ax.fill([-7.5, -7.5, -10, -10], [-7.5, 7.5, 7.5, -7.5], color="red", alpha=0.3, lw=0)
        # ax.fill( [7.5, 7.5, 10, 10], [-7.5, 7.5, 7.5, -7.5], color="red", alpha=0.3, lw=0)

        for i, color in enumerate(colors):
            valid_episode = valid_episodes[i]
            # for valid_episode in valid_episodes:
            trajectories = valid_episode._logs['observations']*10  # steps, n_trajectories, space_dims
            for t in range(trajectories.shape[1]):
                plt.plot(trajectories[:, t, 0], trajectories[:, t, 1], color=color, alpha=0.4)
            for task_dict in valid_episode._logs['infos']:
                goal = task_dict["task"]["goal"]*10
                circle_reward = plt.Circle(goal, 5., color=color, alpha=0.3)
                # circle_goal = plt.Circle(goal, 0.05, color='blue', alpha=0.6)
                ax.add_artist(circle_reward)
                # ax.add_artist(circle_goal)
                break

        # circle_goals = plt.Circle([0, 0], radius=7, fill=False, linewidth=2.0, linestyle='--', color='darkslateblue')
        # ax.add_artist(circle_goals)

        # plt.axis([-10, +10, -10, +10])

        image = plot_to_image(figure)

        with file_writer.as_default():
            tf.summary.image("ValidTrajectories", image, step=batch)
