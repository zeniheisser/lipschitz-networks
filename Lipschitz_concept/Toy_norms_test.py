
import os
import matplotlib
matplotlib.use("Agg")              # no windows: the figures are only saved to files
import matplotlib.pyplot as plt

from Toy_nd_utils import *
from Toy_utils import REPORT_STYLE

# GOAL: investigate the effect of different norms on training of the Lipschitz Network.

QUICK = False       # True: short training, only to check that everything runs
PLOT_DIR = "Plots/Toy_norms"
BATCH_SIZE = 8192   # points per training step


def norm_scan():
    norms = ["l1", "l2", "linf"]
    L_RATIOS = [0.1, 0.3, 0.6, 1.0] if QUICK else np.linspace(0.1, 1.0, 10)
    SEEDS  = [0, 1, 2]
    STEPS = 100 if QUICK else 6000
    POINTS_PER_SIGMA = 4 if QUICK else 8

    # Generate the base Gaussian
    SIGMA = 1.0
    N = 2
    MU = np.zeros(N)
    COVARIANCE = isotropic_cov(sigma=SIGMA, n=N)
    L_base_l2 = gaussian_lipschitz_constant_nd(cov=COVARIANCE, norm="l2") 

    # The grid must contain the Gaussian (5 sigma) and the floor for the smallest L (it gets wider as L gets smaller)
    half_width = max(5 * SIGMA, 1.2 * pyramid_radius(min(L_RATIOS) * L_base_l2, N))
    axes, X = make_grid(half_width, 2 * int(half_width / SIGMA * POINTS_PER_SIGMA) + 1, N)   # odd: peak on the grid
    x = X.reshape(-1, N)                    # the grid points as (number of points, N), the shape the networks take
    base_gaussian = gaussian_nd(X, MU, COVARIANCE)
    FWHM_base = fwhm_equivalent(base_gaussian, axes)

    # Results: the truth values, and per norm one entry per L, each with one value per seed
    results = {"L_base": L_base_l2, "FWHM_base": FWHM_base, "L_ratios": L_RATIOS}
    print(f"L_base (l2) = {L_base_l2:.4f}, FWHM_base = {FWHM_base:.4f}, grid {len(axes[0])}^2 points")

    # Iterate through the norms
    for norm in norms:
        results[norm] = {"relative_error": [], "MSE": [], "FWHM": []}

        # Iterate through the Lipschitz constants
        for L in L_RATIOS:
            L_scaled = L * L_base_l2  # Scale the Lipschitz constant based on the base Gaussian and the ratio
            relative_error_seeds, MSE_seeds, FWHM_seeds = [], [], []

            # Iterate through the seeds: train one network and measure it
            for seed in SEEDS:
                job = dict(kind="lipschitz", seed=seed, steps=STEPS, lr=3e-3, batch_size=BATCH_SIZE, x=x, y=base_gaussian.ravel(),
                           model_kwargs=dict(input_dim=N, hidden_dim=64, output_dim=1, n_groups=8, depth=5,
                                             lipschitz_bound=L_scaled, input_norm=norm))
                prediction = train_one(job)["prediction"].reshape(base_gaussian.shape)

                relative_error_seeds.append(grid_integral((prediction - base_gaussian) ** 2, axes)
                                            / grid_integral(base_gaussian ** 2, axes))
                MSE_seeds.append(np.mean((prediction - base_gaussian) ** 2))
                FWHM_seeds.append(fwhm_equivalent(prediction, axes))

            results[norm]["relative_error"].append(relative_error_seeds)
            results[norm]["MSE"].append(MSE_seeds)
            results[norm]["FWHM"].append(FWHM_seeds)
            print(f"{norm:4s}, L / L_base = {L:.2f}: relative error {np.mean(relative_error_seeds):.3e}, MSE {np.mean(MSE_seeds):3e}"
                  f"FWHM / FWHM_base = {np.mean(FWHM_seeds) / FWHM_base:.3f}")

        # Lists of lists -> arrays of shape (number of L values, number of seeds)
        for name in ["relative_error", "MSE", "FWHM"]:
            results[norm][name] = np.array(results[norm][name])

    return results


#### PLOTS ####
def plot_norm_comparison(results, measure, y_label, file_name, divide_by=1.0, log_y=True):
    """One line per norm: the mean over the seeds against L / L_base, with the spread over the seeds as error bars."""
    with plt.rc_context(REPORT_STYLE):
        fig, ax = plt.subplots(figsize=(3.1, 2.8))
        for norm, label in [("l1", r"$\ell_1$"), ("l2", r"$\ell_2$"), ("linf", r"$\ell_\infty$")]:
            values = results[norm][measure] / divide_by          # shape (number of L values, number of seeds)
            ax.errorbar(results["L_ratios"], values.mean(axis=1), yerr=values.std(axis=1), fmt="o-",
                        lw=1.2, markersize=4, capsize=3, label=label)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(r"$L / L_\mathrm{base}$")
        ax.set_ylabel(y_label)
        ax.legend(frameon=False)
        fig.savefig(f"{PLOT_DIR}/{file_name}")
        plt.close(fig)
#######


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    results = norm_scan()

    plot_norm_comparison(results, "MSE", "MSE", "norms_mse_vs_L.pdf")
    plot_norm_comparison(results, "relative_error", "Relative squared error", "norms_relative_error_vs_L.pdf")
    plot_norm_comparison(results, "FWHM", r"$\mathrm{FWHM} / \mathrm{FWHM}_\mathrm{base}$", "norms_fwhm_vs_L.pdf",
                         divide_by=results["FWHM_base"], log_y=False)
    return 0


if __name__ == "__main__":
    main()