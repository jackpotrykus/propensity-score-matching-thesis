from experiments import experiments
import numpy as np


def main():
    # Set constant hyperparameters
    SIZE1 = 250
    SIZE0 = 750
    P = 5
    NITER_PER_DGM = 30

    # We'll use the same calipers for each experiment too
    CALIPERS = np.round(np.linspace(0.05, 0.5, 10), 2)

    # Setting up experiment 1: caliper vs correlations
    RHOS = np.round(np.linspace(0, 0.8, 9), 2)
    experiment1 = experiments.CaliperExperiment(
        experiment_name="caliper_vs_correlation_big",
        size0=SIZE0,
        size1=SIZE1,
        p=P,
        niter_per_dgm=NITER_PER_DGM,
        calipers=CALIPERS,
        rhos=RHOS,
    )
    experiment1.run()
    experiment1.plot_rho_heatmaps()

    # Setting up experiment 2: caliper vs imbalance
    THETA1S = np.round(np.linspace(0, 1, 11), 2)
    experiment2 = experiments.CaliperExperiment(
        experiment_name="caliper_vs_imbalance_big",
        size0=SIZE0,
        size1=SIZE1,
        p=P,
        niter_per_dgm=NITER_PER_DGM,
        calipers=CALIPERS,
        theta1s=THETA1S,
    )
    experiment2.run()
    experiment2.plot_theta1_heatmaps()


if __name__ == "__main__":
    main()
