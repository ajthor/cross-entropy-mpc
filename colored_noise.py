"""This module contains functions for generating colored noise.

Credit where credit is due: 

This module is a JAX implementation of the `powerlaw_psd_gaussian` function from the 
`colorednoise` package. The original implementation can be found at:

https://github.com/felixpatzelt/colorednoise

The original implementation is licensed under the MIT License.

"""

import jax
import jax.numpy as jnp


def _generate_colored_sample(
    key,
    beta: float,
    shape: tuple,
    fmin: float = 0.0,
):
    """Generates colored noise.

    args:
        key: PRNG key
        beta: Exponent of the power spectrum.
        shape: Shape of the sample. Should be a 3-tuple, (n, N, m), where n is the
            number of samples, N is the number of time steps, and m is the dimension
            of the action space.
        fmin: Low-frequency cutoff.

    returns:
        The colored sample.

    """

    # Jax implementation of powerlaw_psd_gaussian

    # Ensure that shape is a 3-tuple
    assert len(shape) == 3, "Shape should be a 3-tuple."
    n, m, N = shape

    f = jnp.fft.rfftfreq(N)

    if fmin < 0 or fmin > 0.5:
        raise ValueError("fmin must be between 0 and 0.5.")

    fmin = max(fmin, 1.0 / N)

    s_scale = f
    ix = jnp.sum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale = s_scale.at[:ix].set(s_scale[ix])
    s_scale = s_scale ** (-beta / 2)

    w = s_scale[1:].copy()
    w = w.at[-1].set(w[-1] * (1 + (N % 2)) / 2)
    sigma = 2 * jnp.sqrt(jnp.sum(w**2)) / N

    # n_samples = n * N * len(f)

    key, key1, key2 = jax.random.split(key, 3)

    sr = jax.random.normal(key1, shape=(n, m, len(f)))
    sr = sr * s_scale
    sr = sr.reshape((n, m, len(f)))

    si = jax.random.normal(key2, shape=(n, m, len(f)))
    si = si * s_scale
    si = si.reshape((n, m, len(f)))

    if not (N % 2):
        si = si.at[..., -1].set(0)
        sr = sr.at[..., -1].set(sr[..., -1] * jnp.sqrt(2))

    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * jnp.sqrt(2))

    s = sr + 1j * si

    y = jnp.fft.irfft(s, n=N, axis=-1) / sigma

    return y
