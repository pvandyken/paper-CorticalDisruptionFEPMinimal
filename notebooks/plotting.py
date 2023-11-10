import io
import itertools as it

import graph_tool.all as gt
import more_itertools as itx
import nibabel as nb
import numpy as np
import seaborn as sns
import templateflow.api as tflow
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from nilearn.plotting import plot_surf
from statannotations.Annotator import Annotator
from statsmodels.sandbox.stats.multicomp import TukeyHSDResults
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.contrast import ContrastResults
import yaml

from notebooks.graph_tool_utils import (
    encode_features,
    graph_from_weighted_matrix,
    partition_graph,
    rotate_layout,
)


class StatResults:
    def __init__(self):
        self.results = {}

    def add_f(self, label: str, result):
        self.results[label] = self._latex_wrap(self._format(result))

    def add_t(self, label: str, result):
        self.results[label] = self._latex_wrap(self._format_t(result))

    def add_p(self, label: str, result, adj=False, precision=2):
        self.results[label] = self._latex_wrap(self._format_p(result, adj, precision))

    def __repr__(self):
        output = io.StringIO()
        yaml.dump(self.results, output)
        return output.getvalue()

    def _latex_wrap(self, val):
        return f"${val}$"

    def _format_p(self, result, adj=False, precision=2):
        arg = "p_{adj}" if adj else "p"
        predicate = (
            "< 0.001"
            if result < 0.001
            else f"= {np.format_float_positional(result, precision, fractional=False)}"
        )
        return f"{arg} {predicate}"

    def _format(self, result):
        if isinstance(result, RegressionResultsWrapper):
            return (
                f"F({result.df_model:.0f}, {result.df_resid:.0f}) = {result.fvalue:.2f}, "
                f"{self._format_p(result.f_pvalue)}"
            )

    def _format_t(self, result):
        if isinstance(result, ContrastResults):
            return (
                f"t({result.df_denom:.0f}) = {result.tvalue.item():.2f}, "
                f"{self._format_p(result.pvalue)}"
            )


def fig_to_numpy(fig, **kwargs):
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw", **kwargs)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    return img_arr


def plot_on_atlas(
    stat,
    hem=["L", "R"],
    view=["lateral", "medial"],
    facecolor="white",
    transparent=False,
    ax=None,
    vmax=None,
):
    imgs = []
    if not isinstance(hem, str) or not isinstance(view, str):
        for v in itx.always_iterable(view):
            img_row = []
            for h in itx.always_iterable(hem):
                data = plot_on_atlas(
                    stat,
                    h,
                    v,
                    ax=None,
                    vmax=None,
                    facecolor=facecolor,
                    transparent=transparent,
                )
                img_row.append(data)
            imgs.append(np.hstack(img_row))
        return np.vstack(imgs)

    figsize = 5
    if ax is None:
        return_numpy = True
        fig = plt.figure(figsize=(figsize, figsize))
        ax = fig.add_subplot(111, projection="3d")
    else:
        return_numpy = False
    p_ = fig.dpi * figsize
    crop_w = np.s_[int(p_ * 0.25) : int(p_ * 0.733), int(p_ * 0.223) : int(p_ * 0.83)]
    infl = tflow.get(template="fsLR", hemi=hem, density="32k", suffix="inflated")
    lookup = project_to_atlas(stat, hem)
    plot_surf(
        str(infl),
        lookup,
        cmap="plasma",
        # symmetric_cmap=False,
        hemi={"L": "left", "R": "right"}[hem],
        view=view,
        # vmin=0,
        vmax=vmax,
        # bg_on_data=True,
        darkness=0.2,
        axes=ax,
    )
    ax.set_facecolor(facecolor)
    if return_numpy:
        arr = fig_to_numpy(fig, transparent=transparent)[crop_w]
        plt.close(fig)
        return arr
    return ax


def project_to_atlas(data, hem):
    dparc = nb.load(
        f"resources/atlases/atlas-brainnetome210/atlas-brainnetome210_space-fsLR_den-32k_hemi-{hem}.label.gii"
    )
    # Get rid of -1 as an index
    lookup = np.maximum(dparc.get_arrays_from_intent("NIFTI_INTENT_LABEL")[0].data, 0)
    return np.r_[0, data][np.maximum(lookup.data, 0)]


def plot_hierachical_connectome(
    arr, *, ax, atlas, ecmap=cm.plasma, vcmap=cm.tab10, emin=None, emax=None
):
    g = graph_from_weighted_matrix(arr)
    weights = g.ep["weight"]

    features = encode_features(atlas["Lobe"], atlas["hemisphere"])

    state = partition_graph(g, features)
    lobes = features[0]

    tree, *_ = gt.get_hierarchy_tree(state)
    order = tree.own_property(g.degree_property_map("total"))
    # Change the order of hemispheres
    num_lobes = np.unique(lobes).shape[0]
    n_vertices = g.num_vertices()
    order.a[n_vertices : num_lobes * 2 + n_vertices] = np.r_[
        np.r_[:num_lobes][::-1], :num_lobes
    ]
    layout = gt.radial_tree_layout(
        tree, root=tree.vertex(tree.num_vertices() - 1), rel_order=order
    )

    if emin is not None or emax is not None:
        emin = np.min(weights.a) if emin is None else emin
        emax = np.max(weights.a) if emax is None else emax
        range_ = emax - emin
        ecmap = truncate_colormap(
            ecmap,
            max((np.min(weights.a) - emin) / range_, 0),
            min((np.max(weights.a) - emin) / range_, 1),
            256,
        )
        weights.a = np.minimum(weights.a, np.asarray(emax))
        weights.a = np.maximum(weights.a, np.asarray(emin))

    gt.draw_hierarchy(
        state,
        mplfig=ax,
        layout=rotate_layout(layout, -0.499 * np.pi),
        vcmap=vcmap,
        ecmap=ecmap,
        vertex_fill_color=g.new_vertex_property("int", vals=lobes),
        vertex_color=g.new_vertex_property("int", vals=lobes),
        vertex_pen_width=0.01,
        hide=5,
        edge_color=weights,
        edge_pen_width=0.01,
        edge_gradient=[],
    )
    ax.text(
        0.02, 0.5, "L", size=15, weight="bold", color="#303030", transform=ax.transAxes
    )
    ax.text(
        0.95, 0.5, "R", size=15, weight="bold", color="#303030", transform=ax.transAxes
    )


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )


def add_legend(fig, labels, cmap=None, size=None, cmax=None, **kwargs):
    if cmap is None:
        cmap = cm.get_cmap(plt.rcParams["image.cmap"])
    if isinstance(cmap, colors.ListedColormap) and cmax is None:
        cmax = len(cmap.colors)
    colorpoints = colors.Normalize(0, cmax)(np.r_[: len(labels)])
    handles = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            color=cmap(color),
            marker="o",
            label=label,
            markersize=size,
        )
        for label, color in zip(labels, colorpoints)
    ]

    return fig.legend(
        handles=handles,
        **kwargs,
    )


def add_colorbar(
    lower=0,
    upper=1,
    cmap=None,
    cax=None,
    **kwargs,
):
    return plt.colorbar(
        cm.ScalarMappable(colors.Normalize(lower, upper), cmap), cax=cax, **kwargs
    )


def comparison_plot(data, order, ax, y="adj", ylabel="Average FA", significance=None):
    sns.boxplot(
        data=data,
        x="group",
        y=y,
        color="#fafafa",
        order=order,
        boxprops={"facecolor": "#fafafa", "edgecolor": "#505050"},
        showfliers=False,
        width=0.5,
        linewidth=1,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="group",
        y=y,
        order=order,
        ax=ax,
        hue="group",
        legend=False,
        palette="tab10",
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Phenotype")
    if isinstance(significance, TukeyHSDResults):
        significance = {
            pair: significance.pvalues[i]
            for i, pair in enumerate(it.combinations(significance.groupsunique, 2))
            if significance.reject[i]
        }
    if not significance:
        return ax
    annot = Annotator(
        ax,
        data=data,
        x="group",
        y=y,
        order=order,
        pairs=list(significance.keys()),
    )
    annot.configure(test=None).set_pvalues(list(significance.values())).annotate()
    return ax
