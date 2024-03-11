import jax
import jax.numpy as jnp

import gymnasium as gym

from cem import CEMParams
from cem import sample
from cem import update_params
from cem import select_elites
from cem import shift_elites
from cem import shift_params

import tqdm

rng = jax.random.PRNGKey(0)

env = gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="videos/")
proto_env = gym.make("Pendulum-v1", g=9.81)

n_horizon = 30

n_sequences = 20
n_elites = 10
n_old_elites = 5
decay_factor = 1.0

mean = jnp.zeros((n_horizon, env.action_space.shape[0]))
std = jnp.ones((n_horizon, env.action_space.shape[0]))
rng, key = jax.random.split(rng)
elites = jnp.zeros((n_elites, n_horizon, env.action_space.shape[0]))

params = CEMParams(mean=mean, std=std, elites=elites)


def roll_out(env, x0, action_sequences):
    """Roll out the action sequences in the environment."""
    n, N, _ = action_sequences.shape
    trajectories = jnp.zeros((n, N, env.observation_space.shape[0]))
    scores = jnp.zeros(n)
    for i in range(n):
        x = env.reset(seed=0)
        env.state = x0  # Set the initial state
        for j in range(N):
            x, r, *_ = env.step(action_sequences[i, j])
            trajectories = trajectories.at[i, j].set(x)
            scores = scores.at[i].set(scores[i] + r)
    return trajectories, scores


n_sim_horizon = 100
n_iters = 10

x = env.reset(seed=0)
env.start_video_recorder()
env.render()

# Initialize elites to random sequences
rng, params.elites = sample(rng, n=n_elites, params=params)
_, elite_scores = roll_out(proto_env, x, params.elites)
elite_scores = -elite_scores

for t in tqdm.trange(n_sim_horizon):

    # Shift the parameters and repeat the last time step.
    params = shift_params(params)
    params.std = jnp.ones_like(params.std)

    for i in range(n_iters):
        _n_sequences = int(max(n_sequences / decay_factor**i, 2 * n_elites))

        rng, action_sequences = sample(rng, n=_n_sequences, params=params)
        action_sequences = action_sequences.clip(
            env.action_space.low, env.action_space.high
        )

        # Roll out the action sequences and get the scores
        trajectories, scores = roll_out(proto_env, x, action_sequences)

        action_sequences = jnp.concatenate([action_sequences, params.elites], axis=0)
        scores = jnp.concatenate([-scores, elite_scores], axis=0)

        # # Concatenate with previous elites
        # if i == 0:
        #     shifted_elites = shift_elites(rng, params.elites, params=params)
        #     action_sequences = jnp.concatenate(
        #         [action_sequences, shifted_elites], axis=0
        #     )
        # else:
        #     action_sequences = jnp.concatenate(
        #         [action_sequences, params.elites], axis=0
        #     )

        # # If the last iteration, then add mean to the action sequences
        # if i == n_iters - 1:
        #     action_sequences = jnp.concatenate(
        #         [action_sequences, params.mean[None, :, :]], axis=0
        #     )

        # Select the top n elites
        elites, elite_scores = select_elites(action_sequences, scores, n=n_elites)

        tqdm.tqdm.write(f"Elite score: {elite_scores[0]}")

        # Keep n old elites
        old_elites = params.elites[:n_old_elites]
        elites = jnp.concatenate([elites, old_elites], axis=0)

        # Update the parameters
        params = update_params(elites, params=params, theta=1.0)

    # Add the mean to the elites
    elites = jnp.concatenate([elites, params.mean[None, :, :]], axis=0)

    rng, elites = shift_elites(rng, elites, params=params)
    params = update_params(elites, params=params)

    # Choose the best action
    best_action = elites[0, 0]

    # Simulate the environment forward one step
    x, r, *_ = env.step(best_action)
    env.render()


env.close_video_recorder()
env.close()
proto_env.close()
