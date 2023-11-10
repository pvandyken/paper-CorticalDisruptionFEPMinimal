# %%
from __future__ import annotations

from typing import Any, Callable, Iterable, TypeVar
import copy
import itertools as it
import more_itertools as itx
import numpy as np
import networkx as nx
from networkx.algorithms import community
import pandas as pd
from numpy.typing import ArrayLike, NDArray
import functools as ft
from colour import Color
import attrs
import plotly.graph_objects as go

T = TypeVar("T", bound=np.dtype)


class AdjMatrixDim(pd.DataFrame):
    def __getitem__(self, __item: Any):
        series = super().__getitem__(__item)
        if not isinstance(series, pd.Series):
            return series
        return np.array(np.meshgrid(series, series))


def filter_edges(__func: Callable[[nx.Graph, Any, Any], bool], G: nx.Graph):
    _G = G.copy()
    for i, j in _G.edges:
        if not __func(_G, i, j):
            _G.remove_edge(i, j)
    return _G


def np_str_join(
    __delim: str,
    /,
    arrays: NDArray[np.str_] | Iterable[NDArray[np.str_]],
    axis: int | None = None,
) -> NDArray[np.str_] | str:
    if axis is None:
        return __delim.join(itx.collapse(arrays))
    if axis:
        return np_str_join(__delim, np.moveaxis(np.array(arrays), axis, 0), axis=0)
    arrays = iter(arrays)
    try:
        first = next(arrays)
    except StopIteration:
        return np.array([], dtype=np.str_)
    try:
        second = next(arrays)
    except StopIteration:
        return first
    mod = np.char.mod(f"%s{__delim}", first)
    return np_str_join(
        __delim,
        it.chain([np.char.add(mod, second)], arrays),
        axis=0,
    )


def group_outer(__A: ArrayLike, __B: ArrayLike | None = None, /) -> NDArray[Any]:
    if __B is not None:
        return np.dstack(np.meshgrid(__A, __B))
    else:
        return group_outer(__A, __A)


def get_labels(df, fields):
    labels = np.dstack(
        [
            np_str_join(" ➜ ", group_outer(df[field]).astype(str), axis=2)
            for field in fields
        ]
    )
    template = "<br>".join(
        f"<b>{name}:</b> %{{customdata[{i}]}}" for i, name in enumerate(fields)
    )
    return labels, template


def get_names(df):
    return df["Name"] + "_" + df["hemisphere"].str.lower()


def community_sort(adj):
    components = community.greedy_modularity_communities(adj.graph, weight="weight")
    order = list(it.chain.from_iterable(components))
    return adj.with_metadata(adj.metadata.reindex(index=np.array(order)))


def filter_logile(bin: int, num_bins: int = 10):
    def inner(matrix):
        if bin >= num_bins:
            raise ValueError("bin must be less then num_bins")
        masked = np.ma.masked_equal(matrix, 0)
        log = np.ma.log10(masked)
        threshold = 10 ** (((log.max() - log.min()) * bin / num_bins) + log.min())
        return matrix < threshold
    return inner

# %%
@attrs.define
class AdjacencyMatrix:
    raw: NDArray[Any] = attrs.field(converter=np.ma.asarray)
    metadata: pd.DataFrame
    props: dict[str, Any] = attrs.field(factory=dict)
    attrs: dict[str, Any] = attrs.field(factory=dict)

    def __attrs_post_init__(self):
        assert self.raw.shape[0] == self.raw.shape[1]
        assert self.raw.shape[1] == len(self.metadata.index), "Mismatch between length of metadata and shape of data array"

    def __repr__(self):
        return f"AdjacencyMatrix: {self.raw.shape}, {self.attrs}"

    class masks:
        @attrs.frozen
        class _Dim:
            col: str
            dims: list[int]

            def equals(self, __value, /):
                def inner(df):
                    return ft.reduce(
                        np.logical_and,
                        (df[self.col][dim] == __value for dim in self.dims),
                    )

                return inner

        @staticmethod
        def same(column: str):
            def inner(df):
                return df[column][0] == df[column][1]

            return inner

        @staticmethod
        def different(column: str):
            def inner(df):
                return df[column][0] != df[column][1]

            return inner

        @classmethod
        def src(cls, __column: str, /):
            return cls._Dim(__column, [0])

        @classmethod
        def dest(cls, __column: str, /):
            return cls._Dim(__column, [1])

        @classmethod
        def src_and_dest(cls, __column: str, /):
            return cls._Dim(__column, [0, 1])

    @property
    def hems(self):
        return group_outer(self.metadata.sort_index()["hemispheres"])

    def update(self, __new: NDArray[Any]):
        return attrs.evolve(self, raw=__new)

    def with_metadata(self, __metadata: pd.DataFrame):
        return attrs.evolve(self, metadata=__metadata)

    def sort_values(self, *args, **kwargs):
        return self.with_metadata(self.metadata.sort_values(*args, **kwargs))

    def threshold(self, threshold):
        return self.update(np.ma.masked_less(self.raw, threshold))

    def set_index(self, __index, /):
        return self.with_metadata(self.metadata.set_index(__index))

    def mask_where(
        self, __mask_or_func: ArrayLike | Callable[[ArrayLike], ArrayLike], /
    ):
        mask = __mask_or_func(self.raw) if callable(__mask_or_func) else __mask_or_func
        return self.update(np.ma.masked_where(mask, self.raw))

    def mask_diagonal(self):
        return self.mask_where(np.identity(self.raw.shape[0], dtype=bool))

    def mask_equal(self, __value: int, /):
        return self.mask_where(self.raw == __value)

    def filter(self, __func: Callable[[pd.DataFrame], pd.DataFrame]):
        return self.with_metadata(self.metadata[__func(self.metadata)])

    def mask_where_meta(self, __func: Callable[[AdjMatrixDim], NDArray[np.bool_]]):
        mask = copy.copy(self.raw.mask)
        mask[self.ix_] = __func(AdjMatrixDim(self.metadata))
        return self.mask_where(mask)

    @property
    def graph(self):
        df = pd.DataFrame(
            self.filled, columns=self.metadata.index, index=self.metadata.index
        )
        # G = filter_edges(
        #     lambda G, i, j: not np.isnan(G.edges[i, j]["weight"]),
        #     nx.from_pandas_adjacency(df.sort_index())
        # )

        G = nx.from_pandas_adjacency(df.sort_index())
        G.remove_edges_from(
            [
                edge
                for edge in G.edges
                if np.isnan(G.edges[edge]["weight"]) or not G.edges[edge]["weight"]
            ]
        )
        # G.remove_nodes_from([node for node, degree in G.degree if degree == 0])

        for prop, value in self.props.items():
            for edge in G.edges:
                G.edges[edge][prop] = value[edge]
        return G

    @property
    def graphx(self):
        import graph_tool.all as gt
        g = gt.Graph(directed=False)
        arr = self.masked[np.triu_indices_from(self.masked)]
        edges = np.nonzero(arr.filled(0))
        g.add_vertex(self.metadata.index.shape[0])
        g.add_edge_list(np.array(edges).T)
        g.ep["weight"] = g.new_ep("double", vals=arr.filled(0)[edges])
        g.ep["distance"] = g.new_ep("double", vals=(1 / arr).filled(0)[edges])
        return g

    @property
    def filled(self):
        return self.raw.filled(np.NAN)[self.ix_]

    @property
    def masked(self):
        return self.raw[self.ix_]

    @property
    def ix_(self):
        stat_index = self.metadata.index
        return np.ix_(stat_index, stat_index)

    def plot(self, layers=None, labels=None, colorscale=None, **kwargs):
        labels, template = get_labels(self.metadata, labels) if labels else (None, None)

        def get_heatmaps(data):
            colors = map(
                lambda c: Color(f"#{c}"),
                ["21D705", "2DC1E6", "AF2De6", "5b0026", "8aa900"],
            )
            for i, datum in enumerate(data):
                color = next(colors)
                yield go.Heatmap(
                    z=datum,
                    x=get_names(self.metadata),
                    y=get_names(self.metadata),
                    hoverongaps=False,
                    customdata=labels,
                    hovertemplate="<br>".join(
                        [
                            "(%{y}, %{x})",
                            template or "",
                            "T-value: %{z}",
                        ]
                    ),
                    # colorscale=[[0, f"rgba{(*color.rgb, 0.3)}"], [1, color.hex]],
                    colorscale=colorscale,
                    colorbar=dict(x=1.02 + i * 0.08),
                    **kwargs,
                )

        fig = go.Figure()
        if layers:
            fig.add_traces(list(get_heatmaps(layers)))
        else:
            fig.add_traces(list(get_heatmaps([self.filled])))
        fig.update_layout(
            width=1000,
            height=1000,
            # plot_bgcolor="#0d0887",
            yaxis={
                "autorange": "reversed",
            },
        )
        return fig
