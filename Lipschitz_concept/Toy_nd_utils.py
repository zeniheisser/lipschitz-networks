# Utility functions for the n-dimensional Gaussian toy model (Toy_gaussian_2d.ipynb).
# Everything here works for any number of input dimensions n. Functions on a grid are arrays of shape (m_1, ..., m_n):
# the function values at the points of a regular grid made with make_grid.
#
# Which norm? The LipschitzModel networks (first layer normalised with "one-inf", the others with "inf") satisfy
#     |f(x) - f(x')| <= L ||x - x'||_1,   equivalently   max_i |df/dx_i| <= L,
# i.e. every partial derivative is bounded by L separately. Along a diagonal the slope can therefore reach sqrt(n) L.
# All Lipschitz constants, floors and measured slopes below use this norm, unless stated otherwise,
# so that they are the right reference for the networks.
import math

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

from job_runner import run_jobs
from lipschitz_floor import _difference_matrix, _stencil_difference_matrix, _stencil_offsets
from lipschitz_floor import lipschitz_floor_nd  # the floor (best L-Lipschitz fit) on an n-dimensional grid
from models import LipschitzModel, UnconstrainedModel
from Toy_utils import train_mini_batch


##### THE GAUSSIAN #####
def gaussian_nd(x, mean, cov):
    """
    Normalised n-dimensional Gaussian N(mean, cov) evaluated at the points x, shape (..., n).
    Returns an array of shape (...).
    """
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    n = cov.shape[0]
    precision = np.linalg.inv(cov)
    d = x - mean
    exponent = -0.5 * np.einsum("...i,ij,...j->...", d, precision, d)
    return np.exp(exponent) / np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))


def isotropic_cov(sigma, n):
    """Covariance sigma^2 * identity of an isotropic Gaussian in n dimensions."""
    return sigma ** 2 * np.eye(n)


def gaussian_lipschitz_constant_nd(cov, norm="l1"):
    """
    Lipschitz constant of the Gaussian N(mean, cov): the largest value of the dual norm of grad f over all x.
    The mean does not matter, translations do not change the gradients.

    grad f(x) = -P (x - mean) f(x) with P = cov^-1. Writing x - mean = t u with u^T P u = 1, the slope is largest at
    t = 1, which leaves L = exp(-1/2) f_max max_u ||P u||_dual.
        norm = "l1": the networks' norm, dual norm inf,  L = exp(-1/2) f_max max_i sqrt(P_ii)
        norm = "l2": the Euclidean norm,                 L = exp(-1/2) f_max sqrt(largest eigenvalue of P)
    For an isotropic Gaussian both are exp(-1/2) (2 pi)^(-n/2) sigma^-(n+1), which is (sigma^2 sqrt(2 pi e))^-1 in 1D.
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    n = cov.shape[0]
    precision = np.linalg.inv(cov)
    f_max = 1 / np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))
    if norm == "l1":
        factor = np.sqrt(np.max(np.diag(precision)))
    elif norm == "l2":
        factor = np.sqrt(np.max(np.linalg.eigvalsh(precision)))
    else:
        raise ValueError(f"norm ({norm}) must be 'l1' or 'l2'")
    return np.exp(-0.5) * f_max * factor


def isotropic_lipschitz_constant(sigma, n):
    """Lipschitz constant of an isotropic Gaussian with width sigma in n dimensions (the same in the L1 and L2 norm)."""
    sigma = np.asarray(sigma, dtype=float)
    return np.exp(-0.5) * (2 * np.pi) ** (-n / 2) * sigma ** (-(n + 1))


def pyramid_radius(L, n, volume=1.0):
    """
    Rough size of the best L-Lipschitz fit far below L_true: the steepest allowed peak is the pyramid h - L ||x||_1,
    and giving it the same volume as the Gaussian fixes its height, h^(n+1) = volume (n+1)! L^n / 2^n.
    Returns the distance h / L from the peak to the tips of the pyramid (along the axes).
    In 1D this is the tent half-width sqrt(volume / L). Used to choose a grid that is wide enough for the floor.
    """
    h = (volume * math.factorial(n + 1) * L ** n / 2 ** n) ** (1 / (n + 1))
    return h / L


##### GRIDS #####
def make_grid(half_width, points, n, endpoint=True):
    """
    Regular grid on [-half_width, half_width]^n with `points` points along every axis.
    endpoint=False leaves out the upper edge, so the grid is periodic (needed for clean Fourier spectra).
    Returns the list of axes and the points as an array of shape (points, ..., points, n), axis i along dimension i.
    """
    axis = np.linspace(-half_width, half_width, points, endpoint=endpoint)
    axes = [axis] * n
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return axes, X


def grid_spacing(axes):
    """Grid spacing along every axis."""
    return np.array([a[1] - a[0] for a in axes])


def grid_integral(f, axes):
    """Integral of the grid function f over the grid (trapezoidal rule along every axis)."""
    for a in reversed(axes):
        f = np.trapezoid(f, a, axis=-1)
    return f


def grid_gradient(f, axes):
    """Gradient of the grid function f (central differences), as an array of shape (..., n)."""
    if f.ndim == 1:
        return np.gradient(f, axes[0])[..., None]
    return np.stack(np.gradient(f, *axes), axis=-1)


def grid_lipschitz_estimate(f, axes):
    """Largest slope between neighbouring grid points along any axis, i.e. max |df/dx_i| (the networks' L1 norm)."""
    return max(np.max(np.abs(np.diff(f, axis=i))) / (a[1] - a[0]) for i, a in enumerate(axes))


def central_line(f, axis=0):
    """Values of the grid function f along `axis`, through the middle of all the other axes (an exact grid cut)."""
    return f[tuple(slice(None) if i == axis else f.shape[i] // 2 for i in range(f.ndim))]


def profile(axes, f, direction, t, center=None):
    """
    Values of the grid function f along the line center + t * direction (direction is normalised to unit length),
    with linear interpolation between the grid points. Outside the grid the value is NaN.
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    center = np.zeros(len(axes)) if center is None else np.asarray(center, dtype=float)
    interpolator = RegularGridInterpolator(axes, f, bounds_error=False, fill_value=np.nan)
    return interpolator(center + np.outer(t, direction))


##### WIDTHS #####
def half_max_volume(f, axes):
    """Volume (length in 1D, area in 2D) of the region where f is at least half of its maximum."""
    return np.count_nonzero(f >= np.max(f) / 2) * np.prod(grid_spacing(axes))


def fwhm_equivalent(f, axes):
    """
    Width at half maximum in n dimensions: the diameter of the n-ball with the same volume as the region where
    f >= max / 2. This is the FWHM for an isotropic Gaussian in any n, and the normal FWHM in 1D.
    """
    n = f.ndim
    unit_ball = np.pi ** (n / 2) / math.gamma(n / 2 + 1)
    return 2 * (half_max_volume(f, axes) / unit_ball) ** (1 / n)


def fwhm_along(axes, f, direction, center=None, n_points=4001):
    """FWHM along the line through `center` in the given direction, from the interpolated profile of f."""
    t_max = max(a[-1] - a[0] for a in axes)
    t = np.linspace(-t_max, t_max, n_points)
    values = profile(axes, f, direction, t, center)
    above = np.where(values >= np.nanmax(values) / 2)[0]
    if len(above) < 2:
        return 0.0
    return t[above[-1]] - t[above[0]]


##### NETWORKS #####
def evaluate_with_gradient(model, x):
    """Output of the model at the points x, shape (N, n), and its gradient, as numpy arrays of shape (N,) and (N, n)."""
    x_tensor = torch.tensor(np.asarray(x), dtype=torch.float32).requires_grad_(True)
    out = model(x_tensor)
    gradient, = torch.autograd.grad(out.sum(), x_tensor)
    return out.detach().numpy().ravel(), gradient.numpy()


def train_one(job):
    """
    Trains one network with train_mini_batch and evaluates it. job is a dictionary with
        kind: "lipschitz" or "unconstrained",
        model_kwargs: the arguments of LipschitzModel or UnconstrainedModel,
        seed, steps, lr, batch_size,
        x, y: training points and targets, numpy arrays of shape (N, n) and (N,),
        x_eval (optional): where to evaluate the trained network (default: x).
    Returns a dictionary with the prediction, shape (N_eval,), and the gradient, shape (N_eval, n), at x_eval.
    """
    torch.manual_seed(job["seed"])
    model_cls = LipschitzModel if job["kind"] == "lipschitz" else UnconstrainedModel
    model = model_cls(**job["model_kwargs"])
    x = torch.tensor(job["x"], dtype=torch.float32)
    y = torch.tensor(job["y"], dtype=torch.float32).reshape(-1, 1)
    train_mini_batch(model, x, y, job["batch_size"], steps=job["steps"], lr=job["lr"], seed=job["seed"])
    prediction, gradient = evaluate_with_gradient(model, job.get("x_eval", job["x"]))
    return {"prediction": prediction, "gradient": gradient}


def train_many(jobs, n_workers=None):
    """
    Runs train_one for every job in parallel, one process per network, and returns the results in the same order.
    At most job_runner.MAX_JOBS run at a time, and results that are already stored are not trained again.
    """
    return run_jobs(train_one, jobs, n_workers=n_workers, label="Training",
                    depends_on=(train_mini_batch, evaluate_with_gradient, LipschitzModel, UnconstrainedModel))


##### FLOORS #####
def floor_one(job):
    """lipschitz_floor_nd for a dictionary with its arguments. Returns (floor, MSE, info)."""
    return lipschitz_floor_nd(**job, return_info=True)


def floor_many(jobs, n_workers=None):
    """
    Runs floor_one for every job in parallel and returns the results in the same order. A job is a dictionary with y,
    axes, L and optionally the other arguments of lipschitz_floor_nd. Results that are already stored are not solved again.
    """
    return run_jobs(floor_one, jobs, n_workers=n_workers, label="Floors",
                    depends_on=(lipschitz_floor_nd, _difference_matrix, _stencil_difference_matrix, _stencil_offsets))


##### NOISE AND FOURIER SPECTRA #####
def wavevector(modes, length):
    """Angular wave vector of a plane wave with a whole number of periods (one number per axis) across `length`."""
    return 2 * np.pi * np.asarray(modes, dtype=float) / length


def plane_wave_noise(X, amplitudes, wavevectors):
    """Sum of plane waves A sin(k . x) at the points X, shape (..., n), one for each amplitude and wave vector."""
    noise = np.zeros(X.shape[:-1])
    for A, k in zip(amplitudes, wavevectors):
        noise = noise + A * np.sin(X @ np.asarray(k, dtype=float))
    return noise


def plane_wave_slope(A, k):
    """Largest slope of A sin(k . x) in the networks' norm, max_i |df/dx_i| = A max_i |k_i|. In 1D this is A omega."""
    return A * np.max(np.abs(k))


def fourier_modes(axes, f):
    """
    Fourier transform of f on a periodic grid (made with endpoint=False).
    Returns the angular wave vectors, shape (m_1, ..., m_n, n), and the amplitudes |F(k)| / N, so that a plane wave
    A sin(k . x) gives A/2 at k and A/2 at -k.
    """
    coefficients = np.abs(np.fft.fftn(f)) / f.size
    k_axes = [2 * np.pi * np.fft.fftfreq(len(a), d=a[1] - a[0]) for a in axes]
    K = np.stack(np.meshgrid(*k_axes, indexing="ij"), axis=-1)
    return K, coefficients


def mode_amplitude_nd(axes, f, k):
    """Amplitude A of the plane wave A sin(k . x) in f, for a wave vector k that lies on the grid of Fourier modes."""
    K, coefficients = fourier_modes(axes, f)
    index = np.unravel_index(np.argmin(np.sum((K - np.asarray(k)) ** 2, axis=-1)), coefficients.shape)
    return 2 * coefficients[index]


def radial_spectrum(axes, f):
    """
    Amplitude spectrum of f as a function of the length |k| of the wave vector: the n-D version of fourier_spectrum.
    The modes are collected in shells of width 2 pi / (length of the grid), and a shell has the amplitude
    sqrt(2 sum |F(k)/N|^2) over its modes (no factor 2 at k = 0). A single plane wave A sin(k . x) therefore shows up
    as a peak of height A, as in 1D, while noise spread over many directions adds up within the shell.
    Returns the |k| of the shells and their amplitudes.
    """
    K, coefficients = fourier_modes(axes, f)
    dk = 2 * np.pi / max(len(a) * (a[1] - a[0]) for a in axes)
    shell = np.rint(np.linalg.norm(K, axis=-1) / dk).astype(int)
    power = np.bincount(shell.ravel(), weights=(coefficients ** 2).ravel())
    amplitude = np.sqrt(2 * power)
    amplitude[0] = np.sqrt(power[0])
    return dk * np.arange(len(power)), amplitude


def band_power(axes, f, band_lower, band_upper):
    """Sum of |F(k)/N|^2 over the modes with band_lower <= |k| < band_upper, for every band."""
    K, coefficients = fourier_modes(axes, f)
    k_length = np.linalg.norm(K, axis=-1)
    power = coefficients ** 2
    return np.array([np.sum(power[(k_length >= lo) & (k_length < hi)]) for lo, hi in zip(band_lower, band_upper)])
