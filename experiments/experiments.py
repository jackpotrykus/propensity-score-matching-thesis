from dataclasses import dataclass, field
import pathlib
import shutil
from typing import Any, Callable, Iterator, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import seaborn as sns

from experiments.data_generation import ExperimentDataModel
from matching import distance
from matching.graph import MatchingGraph

"""Base directory to save output to"""
OUTPUT_BASE_DIR = (pathlib.Path(__file__).parent / "output").resolve()


def clean_output_base_dir() -> None:
    """Clean (i.e., delete) the contents of the output base directory"""
    if OUTPUT_BASE_DIR.exists():
        shutil.rmtree(OUTPUT_BASE_DIR)


@dataclass
class ExperimentDataLogger:
    """Handles creating directories to save experiment data to

    Attributes
    ----------

    experiment_name: str
        Name of the experiment, which will be the subdirectory of `OUTPUT_BASE_DIR` we save to
    """

    experiment_name: str

    def __post_init__(self) -> None:
        """Create the directories, if they don't exist already"""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> pathlib.Path:
        """Base directory for output"""
        return OUTPUT_BASE_DIR / self.experiment_name

    @property
    def csv_dir(self) -> pathlib.Path:
        """Directory to write CSVs to"""
        return self.output_dir / "csvs"

    @property
    def plot_dir(self) -> pathlib.Path:
        """Directory to write plots to"""
        return self.output_dir / "plots"

    def save_df(self, df: pd.DataFrame, filename: str) -> None:
        """Save `df` to a CSV file. Just a wrapper around `to_csv` method, with some additional defaults

        Parameters
        ----------
        df: pd.DataFrame
            to save
        filename: str
            if not ending in ".csv", this is appended

        Returns
        -------
        None
        """
        ext = ".csv"
        if not filename.endswith(ext):
            filename += ext

        p = self.csv_dir / filename
        df.reset_index().to_csv(p, index=False)

    def save_plot(self, plot_fun: Callable[[], Any], filename: str, dpi: int = 300, **kwargs) -> None:
        """Save a plot to a PNG file. Just a wrapper around `plt.savefig` function, with some additional defaults.

        Note that the default `figsize` is (30, 30), unless overriden in `**kwargs`

        Parameters
        ----------
        plot_fun : Callable[[], Any]
            zero-argument callable producing the plot
        filename : str
            if not ending in ".png", this is appended
        dpi : int
            of the plot, when saving
        **kwargs
            Passed to `matplotlib.pyplot.figure`

        Notes
        -----
        Default `kwargs` are as follows.

        - `figsize: Tuple[int, int] = (30, 30)`


        Returns
        -------
        None
        """

        ext = ".png"
        if not filename.endswith(ext):
            filename += ext

        p = self.plot_dir / filename

        plt.figure(figsize=kwargs.pop("figsize", (30, 30)), **kwargs)
        plot_fun()
        plt.savefig(p, dpi=dpi)


@dataclass
class CaliperExperiment:
    experiment_name: str
    size0: int
    size1: int
    p: int
    calipers: np.ndarray
    rhos: np.ndarray = field(default_factory=lambda: np.array([0], dtype=np.float_))
    theta1s: np.ndarray = field(default_factory=lambda: np.array([1], dtype=np.float_))
    niter_per_dgm: int = field(default=1)

    logger: ExperimentDataLogger = field(init=False)
    rho_caliper_n_match_matrix: np.ndarray = field(init=False)
    rho_caliper_mean_abs_smd_matrix: np.ndarray = field(init=False)
    rho_caliper_max_abs_smd_matrix: np.ndarray = field(init=False)
    theta1_caliper_n_match_matrix: np.ndarray = field(init=False)
    theta1_caliper_mean_abs_smd_matrix: np.ndarray = field(init=False)
    theta1_caliper_max_abs_smd_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Initializes matrices"""
        self.logger = ExperimentDataLogger(self.experiment_name)
        self.rho_caliper_n_match_matrix = np.zeros([self.nrho, self.ncaliper], dtype=np.int_)
        self.rho_caliper_mean_abs_smd_matrix = np.zeros([self.nrho, self.ncaliper], dtype=np.float_)
        self.rho_caliper_max_abs_smd_matrix = np.zeros([self.nrho, self.ncaliper], dtype=np.float_)
        self.theta1_caliper_n_match_matrix = np.zeros([self.ntheta1, self.ncaliper], dtype=np.int_)
        self.theta1_caliper_mean_abs_smd_matrix = np.zeros([self.ntheta1, self.ncaliper], dtype=np.float_)
        self.theta1_caliper_max_abs_smd_matrix = np.zeros([self.ntheta1, self.ncaliper], dtype=np.float_)

    @property
    def nrho(self) -> int:
        """Number of rhos trialed"""
        return self.rhos.shape[0]

    @property
    def ntheta1(self) -> int:
        """Number of theta1s trialed"""
        return self.theta1s.shape[0]

    @property
    def ncaliper(self) -> int:
        """Number of calipers trialed"""
        return self.calipers.shape[0]

    def _update_theta1_matrices(
        self, tdx: int, cdx: int, n_match: int, mean_abs_smd: float, max_abs_smd: float
    ) -> None:
        """Update the theta-caliper matricies. We divide by `self.nrho * self.niter_per_dgm` so that this matrix records the marginal average across all rho"""
        self.theta1_caliper_n_match_matrix[tdx, cdx] += n_match / (self.nrho * self.niter_per_dgm)
        self.theta1_caliper_mean_abs_smd_matrix[tdx, cdx] += mean_abs_smd / (self.nrho * self.niter_per_dgm)
        self.theta1_caliper_max_abs_smd_matrix[tdx, cdx] += max_abs_smd / (self.nrho * self.niter_per_dgm)

    def _update_rho_matrices(self, rdx: int, cdx: int, n_match: int, mean_abs_smd: float, max_abs_smd: float) -> None:
        """Update the rho-caliper matricies. We divide by `self.ntheta1 * self.niter_per_dgm` so that this matrix records the marginal average across all theta1"""
        self.rho_caliper_n_match_matrix[rdx, cdx] += n_match / (self.ntheta1 * self.niter_per_dgm)
        self.rho_caliper_mean_abs_smd_matrix[rdx, cdx] += mean_abs_smd / (self.ntheta1 * self.niter_per_dgm)
        self.rho_caliper_max_abs_smd_matrix[rdx, cdx] += max_abs_smd / (self.ntheta1 * self.niter_per_dgm)

    def _iterate_generated_data(self) -> Iterator[Tuple[Tuple[int, int], Tuple[float, float], ExperimentDataModel]]:
        """Iterate through generated data according to supplied hyperparameters"""
        with tqdm.tqdm(total=self.nrho * self.ntheta1 * self.niter_per_dgm) as pbar:
            for rdx, rho in enumerate(self.rhos):
                for tdx, theta1 in enumerate(self.theta1s):
                    for _ in range(self.niter_per_dgm):
                        data = ExperimentDataModel.generate(
                            size0=self.size1, size1=self.size0, p=self.p, theta0=0, theta1=theta1, rho=rho
                        )
                        yield (rdx, tdx), (rho, theta1), data  # type: ignore
                        pbar.update(1)

    def run(self, plot=True) -> None:
        """Run the experiment"""
        for (rdx, tdx), (rho, theta1), data in self._iterate_generated_data():
            mg = MatchingGraph(data.X, data.z)
            for cdx, caliper in enumerate(self.calipers):
                mg.set_edges(distance=distance.PropensityScore(caliper=caliper)).match(method="optimal")

                try:
                    graph_summary = mg.graph_data.summary().reset_index()
                    graph_subdf = graph_summary.loc[graph_summary["level_1"] == "standardized_difference"]
                except (ValueError, KeyboardInterrupt):
                    # Sometimes a matching cannot be found; we should increase sample size
                    # Print some info about parameters
                    print(f"{self.experiment_name} FAILED: {theta1=}, {rho=}, {caliper=}")
                    continue

                n_match = mg.graph_data.nobs
                standardized_mean_differences = np.asarray(graph_subdf[True].values)
                abs_smds = np.abs(standardized_mean_differences)
                # fmt: off
                self._update_rho_matrices(rdx, cdx, n_match=n_match, mean_abs_smd=np.mean(abs_smds), max_abs_smd=np.max(abs_smds))
                self._update_theta1_matrices(tdx, cdx, n_match=n_match, mean_abs_smd=np.mean(abs_smds), max_abs_smd=np.max(abs_smds))
                # fmt: on

    # def _plot_heatmap_zero_arg(self) -> Callable:

    def plot_theta1_heatmaps(self) -> None:
        def _plot_heatmap_zero_arg(mat: np.ndarray, fmt: str) -> Callable:
            """Return a zero-argument `Callable` that plots a heatmap"""

            def inner() -> None:
                heatmap = sns.heatmap(
                    mat,
                    annot=True,
                    fmt=fmt,
                    xticklabels=self.calipers.astype(str),  # type: ignore
                    yticklabels=self.theta1s.astype(str),  # type: ignore
                )
                heatmap.set_xlabel("Caliper")
                heatmap.set_ylabel("Theta_1 - Theta_0")

            return inner

        figsize = (self.ntheta1 * 2, self.ncaliper * 2)
        # fmt: off
        self.logger.save_plot(_plot_heatmap_zero_arg(self.theta1_caliper_n_match_matrix, fmt="d"), "theta1_caliper_n_match", figsize=figsize)
        self.logger.save_plot(_plot_heatmap_zero_arg(self.theta1_caliper_mean_abs_smd_matrix, fmt="f"), "theta1_caliper_mean_abs_smd", figsize=figsize)
        self.logger.save_plot(_plot_heatmap_zero_arg(self.theta1_caliper_max_abs_smd_matrix, fmt="f"), "theta1_calipermax_abs_smd", figsize=figsize)
        # fmt: on

    def plot_rho_heatmaps(self) -> None:
        def _plot_heatmap_zero_arg(mat: np.ndarray, fmt: str) -> Callable:
            """Return a zero-argument `Callable` that plots a heatmap"""

            def inner() -> None:
                heatmap = sns.heatmap(
                    mat,
                    annot=True,
                    fmt=fmt,
                    xticklabels=self.calipers.astype(str),  # type: ignore
                    yticklabels=self.rhos.astype(str),  # type: ignore
                )
                heatmap.set_xlabel("Caliper")
                heatmap.set_ylabel("Rho")

            return inner

        figsize = (self.nrho * 2, self.ncaliper * 2)
        # fmt: off
        self.logger.save_plot(_plot_heatmap_zero_arg(self.rho_caliper_n_match_matrix, fmt="d"), "rho_caliper_n_match", figsize=figsize)
        self.logger.save_plot(_plot_heatmap_zero_arg(self.rho_caliper_mean_abs_smd_matrix, fmt="f"), "rho_caliper_mean_abs_smd", figsize=figsize)
        self.logger.save_plot(_plot_heatmap_zero_arg(self.rho_caliper_max_abs_smd_matrix, fmt="f"), "rho_calipermax_abs_smd", figsize=figsize)
        # fmt: on
