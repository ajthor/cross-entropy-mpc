import numpy as np

from dataclasses import dataclass

from colored_noise import _generate_colored_sample


@dataclass
class CEMParams:
    """Parameters for the Cross-Entropy Method (CEM)."""

    mean: np.ndarray
    """Mean of the multivariate normal distribution."""

    std: np.ndarray
    """Standard deviation of the multivariate normal distribution."""

    elites: np.ndarray
    """Elite samples from the previous iteration."""


def sample(n: int, params: CEMParams):
    """Sample action sequences from the multivariate normal distribution.

    args:
        rng: PRNG key
        n: Number of sequences to generate.
        params: CEMParams

    returns:
        sample: The sample of action sequences.

    """

    shape = params.mean.shape

    samples = _generate_colored_sample(1, (n, shape[1], shape[0]))
    samples = samples.transpose((0, 2, 1)) * params.std + params.mean

    return samples


def update_params(elites: np.ndarray, params: CEMParams, theta: float = 1.0):
    """Update the parameters of the CEM.

    args:
        elites: The top n elites samples.
        params: CEMParams
        theta: The mixing parameter between 0 and 1, where 0 means no update and 1 means
            full update.

    returns:
        CEMParams

    """

    mean = np.mean(elites, axis=0) * theta + params.mean * (1 - theta)
    std = np.std(elites, axis=0) * theta + params.std * (1 - theta)

    return CEMParams(mean=mean, std=std, elites=elites)


def select_elites(samples: np.ndarray, scores: np.ndarray, n: int = 10):
    """Select the top n elites from the samples based on the scores.

    args:
        samples: Samples to be selected from.
        scores: Scores of the samples.
        n: Number of elites to select.

    returns:
        Tuple of the top n elites samples and their scores.

    """

    # Get the indices of the top n_elites samples
    # NOTE: argsort sorts by increasing order, so we select the first n indices.
    elite_indices = np.argsort(scores)[:n]

    return samples[elite_indices], scores[elite_indices]


def shift_elites(elites: np.ndarray, params: CEMParams):
    """Shift the elites to remove the first action and append a new action.

    args:
        elites: The top n elites samples.
        params: CEMParams

    returns:
        The shifted elites.

    """

    samples = _generate_colored_sample(
        1, (elites.shape[0], elites.shape[2], elites.shape[1])
    )
    samples = samples.transpose((0, 2, 1)) * params.std + params.mean

    # Append the new actions to the elites
    new_actions = samples[:, -1, :]
    shifted_elites = np.concatenate((elites[:, 1:, :], new_actions[:, None, :]), axis=1)

    return shifted_elites


def shift_params(params: CEMParams):
    """Shift the mean and std of the multivariate normal distribution.

    args:
        params: CEMParams

    returns:
        CEMParams

    """

    mean = np.concatenate((params.mean[1:], params.mean[-1:]), axis=0)
    std = np.concatenate((params.std[1:], params.std[-1:]), axis=0)
    std[-1] = 1.0

    return CEMParams(mean=mean, std=std, elites=params.elites)
