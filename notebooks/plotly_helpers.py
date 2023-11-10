import numpy as np
import plotly.graph_objects as go
import itertools as it
import more_itertools as itx
import plotly.express as px
from plotly.subplots import make_subplots

from notebooks.utils import underscore as __
def plotly_grid(
    __fig_gen,
    *conditions,
    vspacing=0.2,
    hspacing=0.1
):
    if len(conditions) > 2:
        raise ValueError("A maximum of 2 conditions may be used")
    rows, cols = __.pipe(
        it.chain(conditions, it.repeat(None)),
        lambda _: itx.take(2, _),
        __.map(lambda _: _ or []),
    )
    # rows, cols = (
    #     __.pipe(conditions)
    #     [it.chain](__@0, it.repeat(None))
    #     [itx.take](2, __@0)
    #     .map(lambda _: _ or [])
    #     .execute()

    # ) # type: ignore
    mappings = {
        "sift2": "# streamlines",
    }
    fig = make_subplots(
        rows = len(rows) or 1,
        cols = len(cols) or 1,
        subplot_titles = tuple(
            __.pipe(
                [
                    col or None,
                    row or None,
                ],
                __.filter(None),
                __.map(lambda _: mappings.get(_, _)),
                __.map(str),
                ": ".join,
            )  # type: ignore
            for row, col in it.product(rows or [None], cols or [None])
        ),
        # row_titles=rows,
        # column_titles=__.pipe(
        #     cols,
        #     __.map(lambda _: mappings.get(_, _)),
        #     list,
        # ),
        horizontal_spacing=hspacing,
        vertical_spacing=min(vspacing, 1 / max((len(rows) or 1) - 1, 1)),
    )
    for (r_i, row), (c_i, col) in __.pipe(
        [rows or [None], cols or [None]],
        __.map(enumerate),
        __.unpack,
        it.product,
    ):
        r_i += 1
        c_i += 1
        if row is not None and col is not None:
            trace = __fig_gen(row, col)
        elif col is not None:
            trace = __fig_gen(col)
        else:
            trace = __fig_gen(row)
        if isinstance(trace, go.Figure):
            fig.add_traces(trace.data, rows=r_i, cols=c_i)
            for annot in trace.layout["annotations"]:
                fig.add_annotation(annot, row=r_i, col=c_i)
        else:
            fig.add_trace(trace, row=r_i, col=c_i)
    return fig


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    return [
        [k*h, f"rgb{tuple(map(np.uint8, np.array(cmap(k*h)[:3])*255))}"]
        for k in range(pl_entries)
    ]