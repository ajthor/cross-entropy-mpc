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
    """Sample from the multivariate normal distribution defined by the mean and std.

    args:
        key: PRNG key
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


def select_elites(n: int, samples: jnp.ndarray, scores: jnp.ndarray):
    """Select the top n elites from the samples based on the scores.

    args:
        n: Number of elites to select.
        samples: Samples to be selected from.
        scores: Scores of the samples.

    returns:
        Tuple of the top n elites samples and their scores.

    """

    # Get the indices of the top n_elites samples
    elite_indices = jnp.argsort(scores)[-n:]

    return samples[elite_indices], scores[elite_indices]


def update_params(elites: jnp.ndarray, params: CEMParams):
    """Update the parameters of the CEM.

    args:
        elites: The top n elites samples.
        params: CEMParams

    returns:
        CEMParams

    """

    mean = jnp.mean(elites.flatten(), axis=0)
    std = jnp.std(elites.flatten(), axis=0)

    return CEMParams(mean, std, elites)


# Test
rng = jax.random.PRNGKey(0)
mean = jnp.zeros((10, 3))
std = jnp.ones((10, 3))
elites = jnp.empty_like(mean)
params = CEMParams(mean=mean, std=std, elites=elites)

n = 100

samples = sample(rng, n=100, params=params)


pass
