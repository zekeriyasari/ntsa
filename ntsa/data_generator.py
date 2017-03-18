import pickle
import numpy as np
from scipy.integrate import odeint
import functools


def lorenz_parametric(x, t, sigma=16., b=4., r=45.92):
    """Lorenz system"""

    xdot = sigma * (x[1] - x[0])
    ydot = x[0] * (r - x[2]) - x[1]
    zdot = x[0] * x[1] - b * x[2]

    return np.array([xdot, ydot, zdot])


def create_data(system, x0, filename, n=5000, ts=0.01, state_index=0, out=False, *args, **kwargs):
    """
    Create lorenz data and writes ot a pickle file.
    :param out: bool,
        if True, return the time series
    :param system: str,
        name of the underlying dynamical system for the time series
    :param n: int,
        number of data point
    :param ts: flaot,
        sampling interval
    :param x0: np.ndarray,
        initial condition
    :param filename: str,
        filename for the data
    :param state_index: int,
        state of the Lorenz system, {0: x, 1: y, 2: z}
    :param args: tuple,
        system args,
    :param kwargs: dict,
        system kwargs,
    :poram out: bool,
        if true, return the time series
    :return: np.ndarray,
        sampled Lorenz system data time series
    """
    """
     Solve for 2n data points and discard the first n
     to avoid transients and to approximately converge to the attractor
    """
    t = np.arange(2 * n) * ts
    # if x0 is None:
    #     x0 = np.random.rand(3)

    # try:
    #     system_ = getattr(dynamics, system)
    # except:
    #     raise ValueError('{} is not among the dynamics '.format(system))

    res = odeint(functools.partial(lorenz_parametric, *args, **kwargs), x0, t)
    data = res[n:, state_index]

    content = dict(
        data=data,
        system=system,
        state_index=state_index,
        ts=ts,
        n=n
    )

    with open(filename, 'wb') as file:
        pickle.dump(content, file)

    if out:
        return data
