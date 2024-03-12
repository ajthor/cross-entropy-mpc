import numpy as np

import gymnasium as gym

from cem import CEMParams
from cem import sample
from cem import update_params
from cem import select_elites
from cem import shift_elites
from cem import shift_params

import tqdm

np.random.seed(0)

env = gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="videos/")
proto_env = gym.make("Pendulum-v1", g=9.81)

n_horizon = 20

n_sequences = 100
n_elites = 10
n_old_elites = 5
decay_factor = 1.1

mean = np.zeros((n_horizon, env.action_space.shape[0]))
std = np.ones((n_horizon, env.action_space.shape[0]))
elites = np.zeros((n_elites, n_horizon, env.action_space.shape[0]))

params = CEMParams(mean=mean, std=std, elites=elites)


def roll_out(env, x0, action_sequences):
    """Roll out the action sequences in the environment."""
    n, N, _ = action_sequences.shape
    trajectories = np.zeros((n, N, env.observation_space.shape[0]))
    scores = np.zeros(n)
    for i in range(n):
        env.reset(seed=0)
        env.env.env.env.state = x0  # Set the initial state
        for j in range(N):
            x, r, *_ = env.step(action_sequences[i, j])
            trajectories[i, j] = x
            scores[i] -= r
    return trajectories, scores


n_sim_horizon = 150
n_iters = 5

env.reset(seed=0)
env.start_video_recorder()
env.render()


for t in tqdm.trange(n_sim_horizon):

    # Shift the parameters and repeat the last time step.
    params = shift_params(params)
    params.std = np.ones_like(params.std)

    x = env.env.env.env.state

    for i in range(n_iters):
        _n_sequences = int(max(n_sequences / decay_factor**i, 2 * n_elites))

        action_sequences = sample(n=_n_sequences, params=params)

        # Concatenate with previous elites
        if i == 0:
            shifted_elites = shift_elites(params.elites, params=params)
            action_sequences = np.concatenate(
                [action_sequences, shifted_elites],
                axis=0,
            )
        else:
            action_sequences = np.concatenate(
                [action_sequences, params.elites],
                axis=0,
            )

        # If the last iteration, then add mean to the action sequences
        if i == n_iters - 1:
            action_sequences = np.concatenate(
                [action_sequences, params.mean[None, :, :]],
                axis=0,
            )

        action_sequences = action_sequences.clip(
            env.action_space.low, env.action_space.high
        )

        # Roll out the action sequences and get the scores
        trajectories, scores = roll_out(proto_env, x, action_sequences)

        # Select the top n_elites samples
        elites, elite_scores = select_elites(action_sequences, scores, n=n_elites)

        # Update the parameters
        params = update_params(elites, params=params, theta=1.0)

        tqdm.tqdm.write(f"Elite score: {elite_scores[0]}")

    # Choose the best action
    best_action = elites[0, 0]

    # Simulate the environment forward one step
    x, r, *_ = env.step(best_action)
    env.render()


env.close_video_recorder()
env.close()
proto_env.close()
