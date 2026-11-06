# Utility functions for the toy example notebook
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

from job_runner import cache_key, load_result, save_result


##### DATASET GENERATION #####
def sine_function_train_data(N, frequency=1.0, seed=6):
    """Generating the sine dataset with noise. Based on the LHCb paper."""
    torch.manual_seed(seed) # Setting seed for reproducibility

    x_train = torch.linspace(-2*np.pi, 2*np.pi, N).unsqueeze(-1)
    
    y_train_error = torch.ones(N).unsqueeze(-1) * 0.01 
    y_train_error[N // 2] = 1
    y_train = torch.sin(frequency * x_train) + torch.randn(N).view(-1, 1) * y_train_error # Adding noise to increase learning

    # Plot data
    x_linspace = torch.linspace(-2*np.pi, 2*np.pi, 2*N).unsqueeze(-1)
    y_linspace = torch.sin(frequency * x_linspace)

    return x_train, y_train, x_linspace, y_linspace

def gaussian_function_data(N, mu, sigma, seed=6, noise_magnitude=0.01, range=(-5, 5)):
    """
    Generate data from a Gaussian function with added noise.

    The range must be wide enough for the best L-Lipschitz fit (a tent with half-width sqrt(area / L)) to come back
    down to zero inside it, otherwise the edges of the range, not L, set the width of the fit.
    """
    torch.manual_seed(seed)  # Setting seed for reproducibility

    x_train = torch.linspace(range[0], range[1], N).unsqueeze(-1)

    # Adding some noise to the Gaussian function to make it more challenging for the model to learn
    train_error = torch.randn(N).view(-1, 1) * noise_magnitude # Sampling N random numbers from a normal distribution
    gaussian = (1/(sigma * torch.sqrt(torch.tensor(2 * np.pi)))) * torch.exp(-0.5 * ((x_train - mu) / sigma) ** 2)
    y_train = gaussian +  train_error  # Adding noise to increase learning

    # Plotting data
    x_linspace = torch.linspace(range[0], range[1], 2*N).unsqueeze(-1)
    y_linspace = (1/(sigma * torch.sqrt(torch.tensor(2 * np.pi)))) * torch.exp(-0.5 * ((x_linspace - mu) / sigma) ** 2)
    return x_train, y_train, x_linspace, y_linspace

def gaussian_function(x, mu, sigma):
    """Calculates the Gaussian function for given x, mu, and sigma."""
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

class BasicDataset(Dataset):
    """A simple dataset class for handling input and target data."""
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

###### LIPSHITZ CALCULATIONS ######
# Lipschitz constant of the gaussian
def gaussian_lipschitz_constant(sigma):
    """
    Calculating the Lipschitz constant of a Gaussian function.
    Do not need to care about the mu as translations do no change the gradients of a function
    """
    sqrt_2_pi_e = np.sqrt(2 * np.pi*np.e)
    sigma_squared = sigma ** 2
    return np.abs(-1/(sigma_squared * sqrt_2_pi_e))


def calculate_lipschitz_constant(model, x_data):
    """Calculates the Lipschitz constants from the prediction of a model. This is done by computing the Jacobian of the models output wrt its input, and calculating its maximum singular value which correspondst to the Lipschitz constant.
    
    NOTE: works for small datasets. Larger datasets should be subsampled to avoid memory issues. Potential local estimates can be done as well.
    """
    model.eval()
    with torch.no_grad():
        # Compute the Jacobian of the model's output with respect to its input
        x_data = x_data.clone().detach().requires_grad_(True)  # Ensure x_data requires gradients
        y_pred = model(x_data)
        jacobian = torch.autograd.functional.jacobian(model, x_data)

        # Calculate the Lipschitz constant as the maximum singular value of the Jacobian
        lipschitz_constant = torch.linalg.norm(jacobian, ord=2, dim=(1, 2)).max().item()

    return lipschitz_constant


#### RESOLUTION FUNCTIONS ####
def FWHM_calculation(x, f):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a function f defined on x.
    Returns the FWHM value.
    """
    half_max = np.max(f) / 2.0
    indices = np.where(f >= half_max)[0]
    if len(indices) < 2:
        return 0.0  # Not enough points to define FWHM
    fwhm = x[indices[-1]] - x[indices[0]]
    return fwhm

#### OPTIMIZING THE FLOOR ####




###### TRAINING ######
def train_full_batch(model, x, y, steps=6000, lr=3e-3, noise_fn=None):
    """
    Plain full-batch training with Adam, a cosine learning rate schedule and MSE loss.
    Used for the small 1D robustness tests, where the whole data set fits in one batch.

    noise_fn (optional): a function without arguments that returns a fresh noise tensor with the shape of y.
        If given, every training step uses the target y + noise_fn(), so the network sees a NEW noise realisation
        at every evaluation instead of one fixed realisation added to the data beforehand. Pass the CLEAN target as y.
    Returns the trained model.

    The trained weights are stored on disk: the same model, starting weights, data and settings are not trained again.
    """
    key = cache_key(train_full_batch, model, x, y, steps, lr)
    stored_state = load_result(key)
    if stored_state is not None:
        model.load_state_dict(stored_state)
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    for _ in range(steps):
        optimizer.zero_grad()
        target = y if noise_fn is None else y + noise_fn()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    save_result(key, model.state_dict())
    return model


def train_mini_batch(model, x, y, batch_size, steps=6000, lr=3e-3, seed=0):
    """
    The same training as train_full_batch (Adam, cosine learning rate schedule, MSE loss), but every step uses batch_size
    random points, drawn with replacement. Used for the 2D grids, where a full batch is slow to train. Small batches
    are not equivalent to a full batch: they copy less of the fine noise, and the difference shrinks as batch_size grows.
    With batch_size or fewer points, every step uses all of them. The points are drawn with their own random numbers
    from seed, so the training does not depend on or change the global random state.
    Returns the trained model. The trained weights are stored on disk, as for train_full_batch.
    """
    key = cache_key(train_mini_batch, model, x, y, steps, lr, batch_size, seed)
    stored_state = load_result(key)
    if stored_state is not None:
        model.load_state_dict(stored_state)
        return model

    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    for _ in range(steps):
        optimizer.zero_grad()
        if batch_size < len(x):
            batch = torch.randint(len(x), (batch_size,), generator=generator)
            loss = ((model(x[batch]) - y[batch]) ** 2).mean()
        else:
            loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    save_result(key, model.state_dict())
    return model


def white_noise_sampler(shape, std, seed=0):
    """
    Returns a function that draws a new white noise tensor (independent Gaussian, standard deviation std) of the given
    shape every time it is called. It has its own random generator, so the noise sequence is reproducible from the seed
    and does not change the random numbers used for the network weights.
    """
    generator = torch.Generator().manual_seed(seed)
    return lambda: std * torch.randn(shape, generator=generator)


def sine_noise_sampler(x, amplitudes, frequencies, seed=0):
    """
    Returns a function that draws new sine-wave noise sum_j A_j sin(omega_j x + phi_j) on the points x (tensor of shape
    (N, 1)) every time it is called, with new random phases phi_j each call. The frequencies and amplitudes stay fixed,
    so every draw has the same spectrum as the fixed sine noise, but the noise averages to zero over many draws.
    """
    generator = torch.Generator().manual_seed(seed)
    amplitudes = torch.tensor(amplitudes, dtype=x.dtype)
    frequencies = torch.tensor(frequencies, dtype=x.dtype)

    def sample():
        phases = 2 * np.pi * torch.rand(len(frequencies), generator=generator, dtype=x.dtype)
        return (amplitudes * torch.sin(x * frequencies + phases)).sum(dim=-1, keepdim=True)
    return sample


###### NOISE AND FOURIER SPECTRA ######
def fourier_spectrum(x, f):
    """
    Calculates the amplitude spectrum of f, sampled on the evenly spaced grid x.
    Returns the angular frequencies omega and the amplitude at each frequency.
    The amplitudes are scaled so that a term A * sin(omega * x) shows up as a peak of height A.
    """
    n = len(f)
    dx = x[1] - x[0]                          # grid spacing

    fourier = np.fft.rfft(f)                  # Fourier transform (only positive frequencies, since f is real)
    amplitude = 2 * np.abs(fourier) / n       # scale so a sine of amplitude A gives a peak of height A
    amplitude[0] = amplitude[0] / 2           # the constant term (omega = 0) should not be doubled

    frequency = np.fft.rfftfreq(n, d=dx)      # frequencies in cycles per unit of x
    omega = 2 * np.pi * frequency             # convert to angular frequency, to match sin(omega * x)
    return omega, amplitude


def noise_removed(prediction, clean, noise):
    """
    Share of the noise energy that the prediction did NOT copy. Works for any kind of noise.
    The error of the prediction is split into a copy of the noise and a rest:
        prediction - clean = copied * noise + rest,   copied = sum((prediction - clean) * noise) / sum(noise^2)
    copied = 1 means the network reproduced the noise exactly, copied = 0 means none of it.
    The noise energy that is kept is copied^2, so noise removed = 1 - copied^2.
    """
    copied = np.sum((prediction - clean) * noise) / np.sum(noise ** 2)
    return 1 - copied ** 2


###### PLOTTING (report format) ######
# One style for all report figures: the same text size everywhere, a closed black frame, no grid, no legend box
REPORT_STYLE = {
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.grid": False, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.6, "figure.dpi": 150, "savefig.bbox": "tight",
}

