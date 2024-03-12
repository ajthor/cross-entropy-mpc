"""This module contains functions for generating colored noise.

Credit where credit is due: 

This module is a JAX implementation of the `powerlaw_psd_gaussian` function from the 
`colorednoise` package. The original implementation can be found at:

https://github.com/felixpatzelt/colorednoise

The original implementation is licensed under the MIT License.

"""

import numpy as np


def _generate_colored_sample(
    beta: float,
    shape: tuple,
    fmin: float = 0.0,
):
    """Generates colored noise.

    args:
        beta: Exponent of the power spectrum.
        shape: Shape of the sample. Should be a 3-tuple, (n, N, m), where n is the
            number of samples, N is the number of time steps, and m is the dimension
            of the action space.
        fmin: Low-frequency cutoff.

    returns:
        The colored sample.

    """

    # Ensure that shape is a 3-tuple
    assert len(shape) == 3, "Shape should be a 3-tuple."
    n, m, N = shape

    f = np.fft.rfftfreq(N)

    if fmin < 0 or fmin > 0.5:
        raise ValueError("fmin must be between 0 and 0.5.")

    fmin = max(fmin, 1.0 / N)

    s_scale = f
    ix = np.sum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-beta / 2)

    w = s_scale[1:].copy()
    w[-1] = w[-1] * (1 + (N % 2)) / 2
    sigma = 2 * np.sqrt(np.sum(w**2)) / N

    sr = np.random.normal(size=(n, m, len(f)))
    sr = sr * s_scale
    sr = sr.reshape((n, m, len(f)))

    si = np.random.normal(size=(n, m, len(f)))
    si = si * s_scale
    si = si.reshape((n, m, len(f)))

    if not (N % 2):
        si[..., -1] = 0
        sr[..., -1] = sr[..., -1] * np.sqrt(2)

    si[..., 0] = 0
    sr[..., 0] = sr[..., 0] * np.sqrt(2)

    s = sr + 1j * si

    y = np.fft.irfft(s, n=N, axis=-1) / sigma

    return y
