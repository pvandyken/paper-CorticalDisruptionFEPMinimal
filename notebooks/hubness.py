import numpy as np
from scipy.stats import rankdata
import seaborn as sns


def property_rank(ds, columns=[], inverse_columns=[], dim="node"):
    def rank(ds, column, inverse=False):
        da = ds[column]
        data = da.data
        ax = list(da.dims).index(dim)
        if inverse:
            data = 1 / (data + 1)
        ranked = rankdata(data, axis=ax) / da.shape[ax]
        return ds.assign({column + "_rank": (da.dims, ranked)})

    for column in columns:
        ds = rank(ds, column)
    for column in inverse_columns:
        ds = rank(ds, column, inverse=True)
    return ds


def hubness(ds, dim="node"):
    cols = ["betweenness", "degree", "clust_coeff"]
    inv_cols = ["path_length"]
    return property_rank(ds, columns=cols, inverse_columns=inv_cols, dim=dim).assign(
        hubness=lambda ds: (
            ds[cols[0] + "_rank"].dims,
            np.mean([ds[s + "_rank"] for s in cols + inv_cols], axis=0),
        )
    )


def plot_hubness_ranks(
    data,
    *,
    x="node",
    hue="hubness",
    y="group",
    ax=None,
    cbar=True,
    cbar_ax=None,
    order=None,
    sort_col=None,
    sort_by=None,
):
    sort_col = sort_col or hue
    if order is not None:
        data = data.sel({y: order})
        first = order[0]
    else:
        first = data[y][0]
    if sort_by is None:
        sort_by = data.sel({y: first})[sort_col]
    elif callable(sort_by):
        sort_by = sort_by(data[sort_col], axis=data[sort_col].get_axis_num(y))
    ax = sns.heatmap(
        data.sortby(sort_by)[hue].transpose(y, x).to_pandas(),
        ax=ax,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cmap="plasma",
        yticklabels=1,
        vmin=0,
        vmax=1,
    )
    ax.tick_params(bottom=False, left=False, labelrotation=0)
    ax.get_xaxis().set_visible(False)
    ax.set(ylabel=None)

    return ax
