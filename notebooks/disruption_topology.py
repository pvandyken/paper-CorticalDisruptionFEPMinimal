from typing import Literal

import dask.array as da
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike, NDArray


def get_permuter(nbs: NDArray[np.float_], subgraph, rng):
    @da.as_gufunc(signature="()->()", output_dtypes=float, vectorize=True)
    def inner(i):
        rand = rng.choice(nbs.shape[0], i)
        return np.mean(np.array(nbs)[_get_selector(rand, subgraph)] > 0)

    return inner


def _get_selector(rois, subgraph):
    if subgraph == "open":
        return np.s_[rois]
    elif subgraph == "closed":
        return np.s_[np.ix_(rois, rois)]
    raise ValueError(f"Invalid variant: {subgraph}")


def get_disruption_by_hubness(
    hubness: ArrayLike,
    nbs: ArrayLike,
    upper: float = 0.90,
    lower: float = 0.1,
    res: float = 0.02,
    kernel: float | Literal["upper", "lower"] = 0.05,
    subgraph: Literal["open", "closed"] = "open",
    seed: int = 2,
    permutations: int = 1000,
    alpha: float = 0.05,
    normalize: bool = True,
):
    nbs = np.asarray(nbs)
    hubness = np.asarray(hubness)
    rng = np.random.default_rng(seed)
    if not np.all(nbs == nbs.T):
        raise ValueError("nbs matrix must be diagonally symmetric")
    vals = np.r_[lower : upper + res : res]
    if isinstance(kernel, (float, int)):
        ls = vals - kernel / 2
        rs = vals + kernel / 2
    elif kernel == "upper":
        ls = vals
        rs = np.full_like(ls, 1)
    elif kernel == "lower":
        rs = vals
        ls = np.full_like(rs, 0)
    else:
        raise ValueError("kernel must be a float or 'upper' or 'lower'")
    counts = np.empty_like(ls, dtype=np.int_)
    empericals = np.empty_like(ls)
    for i in range(ls.shape[0]):
        selected = (hubness > ls[i]) & (hubness < rs[i])
        counts[i] = selected.sum()
        empericals[i] = np.mean(np.array(nbs)[_get_selector(selected, subgraph)] > 0)

    permuter = get_permuter(nbs, subgraph, rng)
    randoms = permuter(
        da.from_array(np.broadcast_to(counts, (permutations, counts.shape[0])))
    ).compute(scheduler="processes")
    mean_distr = np.mean(randoms, axis=0) if normalize else 1

    ci = np.percentile(randoms, [alpha / 2, 100 - alpha / 2], axis=0) / mean_distr
    return pd.DataFrame(
        {
            "x": vals,
            "y": empericals / mean_distr,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
        }
    )


def plot_disruption_topology(data, ax, xlabel=None, legend: bool = True):
    ax.fill_between(
        data["x"],
        data["ci_lower"],
        data["ci_upper"],
        alpha=0.2,
        color="r",
        linewidth=0,
    )
    sns.lineplot(
        data=data,
        x="x",
        y="y",
        ax=ax,
    )
    # ax.axhline(y=1, color="r", linestyle="--")
    ax.set_ylabel("Norm. disruption", fontsize=10, weight="normal")
    ax.set_xlabel("Hubness" if xlabel is None else xlabel, fontsize=10, weight="normal")
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_tick_params(labelsize=8)
    if legend:
        ax.legend(
            loc="upper right",
            labels=["95% CI", "Empirical"],
            fontsize=8,
        )
