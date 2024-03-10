import jax
import jax.numpy as jnp

import chex

from colored_noise import _generate_colored_sample


@chex.dataclass
class CEMParams:
    """Parameters for the Cross-Entropy Method (CEM)."""

    mean: jnp.ndarray
    """Mean of the multivariate normal distribution."""

    std: jnp.ndarray
    """Standard deviation of the multivariate normal distribution."""

    elites: jnp.ndarray
    """Elite samples from the previous iteration."""


def sample(key, n: int, params: CEMParams):
    """Sample action sequences from the multivariate normal distribution.

    args:
        key: PRNG key
        n: Number of sequences to generate.
        params: CEMParams

    returns:
        sample: The sample of action sequences.

    """

    shape = params.mean.shape

    samples = (
        _generate_colored_sample(
            key,
            1,
            (n, shape[1], shape[0]),
        ).transpose((0, 2, 1))
        * params.std
        + params.mean
    )

    return samples


def update_params(elites: jnp.ndarray, params: CEMParams, theta: float = 1):
    """Update the parameters of the CEM.

    args:
        elites: The top n elites samples.
        params: CEMParams
        theta: The mixing parameter between 0 and 1, where 0 means no update and 1 means
            full update.

    returns:
        CEMParams

    """

    if theta < 1:
        mean = jnp.mean(elites, axis=0) * theta + params.mean * (1 - theta)
        std = jnp.std(elites, axis=0) * theta + params.std * (1 - theta)

    else:
        mean = jnp.mean(elites, axis=0)
        std = jnp.std(elites, axis=0)

    return CEMParams(mean=mean, std=std, elites=elites)


def select_elites(samples: jnp.ndarray, scores: jnp.ndarray, n: int = 10):
    """Select the top n elites from the samples based on the scores.

    args:
        samples: Samples to be selected from.
        scores: Scores of the samples.
        n: Number of elites to select.

    returns:
        Tuple of the top n elites samples and their scores.

    """

    # Get the indices of the top n_elites samples
    # NOTE: jnp.argsort sorts by increasing order, so we select the first n indices.
    elite_indices = jnp.argsort(scores)[:n]

    return samples[elite_indices], scores[elite_indices]


def shift_elites(key, elites: jnp.ndarray, params: CEMParams):
    """Shift the elites to remove the first action and append a new action.

    args:
        key: PRNG key
        elites: The top n elites samples.
        params: CEMParams

    returns:
        The shifted elites.

    """

    samples = (
        _generate_colored_sample(
            key,
            1,
            (elites.shape[0], elites.shape[2], elites.shape[1]),
        ).transpose((0, 2, 1))
        * params.std
        + params.mean
    )

    # Append the new actions to the elites
    new_actions = samples[:, -1, :]
    shifted_elites = jnp.concatenate(
        (elites[:, 1:, :], new_actions[:, None, :]), axis=1
    )

    return shifted_elites
