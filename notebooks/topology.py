import graph_tool.all as gt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, permutation_test

from notebooks.plotting import comparison_plot


def compute_kendalltau_distances(data):
    g = gt.Graph(directed=False)
    g = gt.complete_graph(len(data["subject"]))

    vids = g.new_vp("string", data["subject"].data)

    kt = g.new_edge_property("float")
    for a, b, i in g.iter_edges([g.edge_index]):
        kt.a[i] = kendalltau(
            data.sel(subject=vids[a]), data.sel(subject=vids[b])
        ).correlation
    g.ep["kt"] = kt
    g.vp["vids"] = vids
    return g


def compare_hubness_intergroup(x, y, g, seed=2):
    def subgroup_edges(vertices):
        con = g.new_edge_property("bool", val=False)
        con.a[g.vfilt[vertices].eindex] = True
        return con

    def stat(a, b):
        a_edges = subgroup_edges(a)
        b_edges = subgroup_edges(b)
        union = subgroup_edges(np.union1d(a, b))
        intersection = subgroup_edges(np.intersect1d(a, b))
        sub = gt.GraphView(g, efilt=union.a ^ intersection.a ^ a_edges.a ^ b_edges.a)
        return np.mean(sub.ep["kt"].ma)

    rng = np.random.default_rng(seed)
    x_ids = np.nonzero(np.isin(g.v["vids"], x))[0]
    y_ids = np.nonzero(np.isin(g.v["vids"], y))[0]
    return permutation_test(
        (x_ids, y_ids),
        stat,
        n_resamples=10000,
        random_state=rng,
        axis=0,
    )

def compare_hubness_global(x, g, seed=2):
    all_edges = np.r_[:g.num_vertices()]

    def subgroup_edges(vertices):
        con = g.new_edge_property("bool", val=False)
        con.a[g.vfilt[vertices].eindex] = True
        return con

    def stat(a, b):
        a_edges = subgroup_edges(np.setdiff1d(a, all_edges, assume_unique=True))
        b_edges = subgroup_edges(np.setdiff1d(all_edges, a, assume_unique=True))
        union = subgroup_edges(np.union1d(a, all_edges))
        intersection = subgroup_edges(np.intersect1d(a, all_edges, assume_unique=True))
        sub = gt.GraphView(g, efilt=union.a ^ intersection.a ^ a_edges.a ^ b_edges.a)
        return np.mean(sub.ep["kt"].fa)

    rng = np.random.default_rng(seed)
    x_ids = np.nonzero(np.isin(g.v["vids"], x))[0]
    y_ids = np.nonzero(~np.isin(g.v["vids"], x))[0]
    return permutation_test(
        (x_ids, y_ids),
        stat,
        n_resamples=10000,
        random_state=rng,
        axis=0,
    )


def compare_hubness_intragroup(x, y, g, seed=2):
    def get_average_subgroup_kt(subjects):
        return np.mean(g.vfilt[subjects].e["kt"])

    def stat(x, y):
        return get_average_subgroup_kt(x) - get_average_subgroup_kt(y)

    rng = np.random.default_rng(seed)
    vids = g.vp["vids"].get_2d_array([0]).flatten()
    x_ids = np.nonzero(np.isin(vids, x))[0]
    y_ids = np.setdiff1d(np.nonzero(np.isin(vids, y))[0], x_ids)
    return permutation_test(
        (x_ids, y_ids),
        stat,
        n_resamples=10000,
        random_state=rng,
        axis=0,
    )


def plot_intragroup_difference(
    g, da, ax=None, title=None, order=None, significance=None
):
    degr = g.new_vp("double", val=0)
    for _, group in da.groupby("group"):
        x_ids = np.isin(g.v["vids"], group["subject"])
        filt = g.vfilt[x_ids]
        degr.a += (
            filt.degree_property_map("total", filt.e["kt"]).a / filt.num_vertices()
        )
    data = da.merge(
        pd.Series(
            np.array(degr), index=pd.Index(g.v["vids"], name="subject"), name="degr"
        ).to_xarray(),
        join="inner",
    )
    df = data[["degr", "group"]].to_dataframe()
    print(df)
    ax = comparison_plot(
        data=df,
        y="degr",
        order=order or df["group"].unique(),
        ax=ax,
        significance=significance,
    )
    # sns.move_legend(ax, "upper left", fontsize=12, title="Group", title_fontsize=12)

    # ax.get_yaxis().set_visible(False)
    ax.set_ylabel("Intragroup Similarity", size=11)
    ax.set_title(title, size=12, weight="bold")
    return ax


def plot_permutation_result(
    result, ax=None, xdelta=0, title=None, alpha=0.05, significance=None
):
    ax = sns.kdeplot(result.null_distribution, color="red", ax=ax, alpha=0.25)
    l1 = ax.lines[0]

    x1 = l1.get_xydata()[:, 0]
    y1 = l1.get_xydata()[:, 1]

    ax.fill_between(x1, y1, color="red", alpha=0.2)
    ylim = ax.get_ylim()
    ax.vlines(result.statistic, ylim[0], ylim[1] / 2)
    ax.text(
        result.statistic + xdelta,
        ylim[1] * 10 / 19,
        f"p = {result.pvalue:.2}",
        horizontalalignment="center",
        weight="bold" if result.pvalue <= alpha else "normal",
    )
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Similarity", size=11)
    ax.set_title(title, size=12, weight="bold")
