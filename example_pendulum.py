import jax
import jax.numpy as jnp

import gymnasium as gym

from cem import CEMParams
from cem import sample
from cem import update_params
from cem import select_elites
from cem import shift_elites

rng = jax.random.PRNGKey(0)

env = gym.make("Pendulum-v1", g=9.81, render_mode="human")
proto_env = gym.make("Pendulum-v1", g=9.81)

n_horizon = 30

n_sequences = 20
n_elites = 10
n_old_elites = 5
mean = jnp.zeros((n_horizon, env.action_space.shape[0]))
std = jnp.ones((n_horizon, env.action_space.shape[0]))
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
env.render()

for t in range(n_sim_horizon):

    for i in range(n_iters):
        action_sequences = sample(rng, n=n_sequences, params=params)
        action_sequences = action_sequences.clip(
            env.action_space.low, env.action_space.high
        )

        # Concatenate with previous elites
        action_sequences = jnp.concatenate([action_sequences, params.elites], axis=0)

        # Roll out the action sequences and get the scores
        trajectories, scores = roll_out(proto_env, x, action_sequences)

        # Select the top n elites
        elites, elite_scores = select_elites(action_sequences, -scores, n=n_elites)

        # Keep n old elites
        old_elites = params.elites[:n_old_elites]
        elites = jnp.concatenate([elites, old_elites], axis=0)

        # Update the parameters
        params = update_params(elites, params=params, theta=0.1)

    # Choose the best action
    best_action = elites[0, 0]

    # Simulate the environment forward one step
    x, r, *_ = env.step(best_action)
    env.render()

    # Shift the elites
    elites = shift_elites(rng, elites, params=params)
    params = update_params(elites, params=params)

env.close()
proto_env.close()
