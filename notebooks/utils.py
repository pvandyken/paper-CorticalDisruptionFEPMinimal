import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from pathlib import Path
import ipywidgets as widgets
import xarray as xr
import pickle
import itertools as it
import copy
from typing import Any
import shutil
from numpy.typing import ArrayLike


def symmetrize(arr):
    result = copy.copy(np.array(arr))
    result = result + np.tril(result.T)
    return result


def sort_like(arr: ArrayLike, ref: ArrayLike):
    A, ids = np.unique(arr, return_inverse=True)
    _, order = np.unique(
        np.hstack((A, ref)),
        return_inverse=True,
    )
    a_order = order[: A.size]
    v_order = order[-A.size :]
    A[np.argsort(a_order)][v_order]
    return arr[np.argsort(v_order[a_order][ids])]


def circle(radius, n):
    points = np.r_[: 2 * np.pi : complex(n)]
    return np.c_[np.cos(points) * radius, np.sin(points) * radius]


def filter_logile(matrix, bin: int, num_bins: int = 10):
    if bin >= num_bins:
        raise ValueError("bin must be less then num_bins")
    masked = np.ma.masked_equal(matrix, 0)
    log = np.ma.log10(masked)
    threshold = ((log.max() - log.min()) * bin / num_bins) + log.min()
    cp = copy.deepcopy(matrix)
    cp[log <= threshold] = 0
    return cp


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def get_lut(path):
    with open(path) as f:
        lines = [line.strip().split() for line in f.readlines()]
    return {int(key): val for key, val in zip(*list(zip(*lines))[0:2])}


def lut_label(data, path):
    lut = get_lut(path)
    return pd.DataFrame(data).rename(index=lut, columns=lut)


def concat_product(func, *iters, dims, **kwargs):
    def recurse(func, dims, given_args, iters):
        if len(iters) == 1:
            return xr.concat(
                [func(*given_args, val, **kwargs) for val in iters[0]], dim=dims[0]
            )
        return xr.concat(
            [
                recurse(func, dims[1:], given_args + [val], iters[1:])
                for val in iters[0]
            ],
            dim=dims[0],
        )

    return recurse(func, dims, [], iters)


def titleize(label):
    return label.replace("_", " ").capitalize()


def error_line(df, x=None, y=None, color=None, err=None, **kwargs):
    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        **kwargs,
    )
    for trace in fig.data:
        ddf = df[df[color] == trace["name"]] if color else df
        fig.add_traces(
            [
                go.Scatter(
                    x=ddf[x],
                    y=ddf[y] + ddf[err],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    x=ddf[x],
                    y=ddf[y] - ddf[err],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f'rgba{(*hex_to_rgb(trace["line"]["color"]), 0.2)}',
                    showlegend=False,
                ),
            ]
        )
    return fig


def distribution_plot(df, x, y: str):
    std_col = y + "_std"
    grouper = df.groupby(["category", x])
    ddf = (
        pd.concat(
            [
                grouper.mean(),
                grouper.std().rename({y: std_col}, axis="columns"),
            ],
            axis=1,
        )
        .reset_index()
        .set_index("category")
    )
    fig = px.line(
        ddf,
        x=x,
        y=y,
        color=ddf.index,
        width=800,
        height=600,
        labels={
            "x": "Nodes sorted by increasing degree",
            "node_size": "Node Size (# triangles)",
            "betweenness": "Betweenness",
            "degree": "Degree",
        },
        title=f"{titleize(y)} distribution",
    )
    buttons = []
    num_traces = len(ddf.index.unique())
    for i, cat in enumerate(ddf.index.unique()):
        fig.add_traces(
            [
                go.Scatter(
                    x=ddf.loc[cat, x],
                    y=ddf.loc[cat, std_col] + ddf.loc[cat, y],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    x=ddf.loc[cat, x],
                    y=ddf.loc[cat, y] - ddf.loc[cat, std_col],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=f"rgba{(*hex_to_rgb(px.colors.qualitative.Plotly[i]), 0.2)}",
                    showlegend=False,
                ),
            ]
        )
        buttons.append(
            {
                "method": "restyle",
                "visible": True,
                "label": cat,
                "args": [
                    {"visible": False},
                    [i, num_traces + i * 2, num_traces + i * 2 + 1],
                ],
                "args2": [
                    {
                        "visible": True,
                    },
                    [i, num_traces + i * 2, num_traces + i * 2 + 1],
                ],
            }
        )
    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1,
                y=-0.2,
                showactive=True,
                buttons=buttons,
            )
        ],
    )
    return fig


def figures_to_html(figs, filename="dashboard.html"):
    Path(filename).parent.mkdir(exist_ok=True)
    with open(filename, "w") as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split("<body>")[1].split("</body>")[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")


def plotly_tabulate(figs, conditions=None):
    if callable(figs) and conditions:
        figs = [figs(condition) for condition in conditions]
        titles = conditions
    else:
        figs = list(figs)
        titles = [fig.layout["title"]["text"] for fig in figs]
    tab = widgets.Tab()

    def construct(fig):
        if isinstance(fig, go.Figure):
            return go.FigureWidget(fig)
        return fig

    tab.children = [construct(fig) for fig in figs]
    for i, title in enumerate(titles):
        tab.set_title(i, title)
    return tab


class NbCache:
    def __init__(self, *indices: str, root="."):
        self.indicies = indices
        self.root = root

    def __call__(self, name, reset_cache=False):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = (self.cache_dir / name).with_suffix(".pyc")

        def do_cache(func, *args, **kwargs):
            if cache_file.exists() and not reset_cache:
                with cache_file.open("rb") as f:
                    return pickle.load(f)
            result = func(*args, **kwargs)
            with cache_file.open("wb") as f:
                pickle.dump(result, f)
            return result

        def wrapper(func) -> Any:
            def inner(*args, **kwargs):
                return do_cache(func, *args, **kwargs)

            return inner

        return wrapper

    @property
    def cache_dir(self):
        return Path(self.root, ".ipynb_cache", *self.indicies)

    def migrate(self, new: "NbCache", dry: bool = False):
        new.cache_dir.mkdir(parents=True, exist_ok=True)
        for path in self.cache_dir.iterdir():
            if path.is_dir():
                continue
            dest = new.cache_dir / path.name
            if dry:
                print(path, "->", dest)
                continue
            shutil.move(str(path), dest)
