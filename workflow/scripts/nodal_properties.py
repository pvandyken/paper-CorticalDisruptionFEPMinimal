from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
import multiprocessing as mp
import attrs
from notebooks.adjacency_matrix import AdjacencyMatrix
from snakeboost import snakemake_args


class Atlas:
    bn246 = pd.read_csv("resources/atlases/atlas-brainnetome246/labels.tsv", sep="\t")


def filter_logile(bin: int, num_bins: int = 10):
    def inner(matrix):
        if bin >= num_bins:
            raise ValueError("bin must be less then num_bins")
        masked = np.ma.masked_equal(matrix, 0)
        log = np.ma.log10(masked)
        threshold = 10 ** (((log.max() - log.min()) * bin / num_bins) + log.min())
        return matrix < threshold

    return inner


@attrs.frozen
class Subject:
    adj: AdjacencyMatrix
    metadata: dict[str, str]
    threshold: int

    @classmethod
    def from_path(cls, path: Path, threshold, wildcards):
        adj = (
            AdjacencyMatrix(
                raw=np.genfromtxt(path, delimiter=",")[1:, 1:],
                metadata=Atlas.bn246,
            )
            .mask_diagonal()
            .mask_equal(0)
            .mask_where(filter_logile(threshold))
        )
        adj.props["distance"] = np.ma.filled(1 / adj.raw, np.NaN)
        return cls(
            adj=adj,
            metadata=wildcards,
            threshold=threshold
        )


def graph_params(subj):
    subj_rows = []
    G = subj.adj.graph
    b = nx.betweenness_centrality(G, weight="distance")
    for node in G:
        subj_rows.append(
            {
                **subj.metadata,
                "node": node,
                "threshold": subj.threshold,
                "degree": G.degree(weight="weight")[node],
                "clust_coeff": nx.clustering(G, nodes=node, weight="weight"),
                "path_length": np.mean(
                    list(
                        nx.shortest_path_length(
                            G, source=node, weight="distance"
                        ).values()
                    )
                ),
                "betweenness": b[node],
            }
        )
    return pd.DataFrame(subj_rows)




def main():
    args = snakemake_args(
        input={"graph": "in"},
        output=["out"],
    )

    if not isinstance(args.input, dict) and (
        "graph" not in args.input or "metadata" not in args.input
    ):
        raise TypeError("Inputs must be a dict with graph and metadata")
    if isinstance(args.output, dict) or len(args.output) != 1:
        raise TypeError("Outputs must be specified as a single item")
    if not isinstance(args.wildcards, dict):
        raise TypeError("wildcards must be a dict")

    graphs = (
        Subject.from_path(
            args.input['graph'],
            threshold=threshold,
            wildcards=args.wildcards,
        )
        for threshold in range(1)
    )
    with mp.Pool() as pool:
        params = pd.concat(pool.map(graph_params, graphs))
    params.to_csv(args.output[0], sep="\t", index=False)


if __name__ == "__main__":
    main()
