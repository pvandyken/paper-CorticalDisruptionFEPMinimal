import graph_tool.all as gt
import numpy as np

def _weighted_rich_club(G):
    w_ranked = np.sort(G.ep["weight"].a)[::-1]
    degree = G.degree_property_map("total", weight=G.ep["weight"]).a
    k_values = np.unique(np.sort(degree))
    results = np.empty(len(k_values))
    for rank, degree_thresh in enumerate(k_values):
        sG = gt.GraphView(G, vfilt=degree > degree_thresh)
        rich_weight = sG.get_edges(eprops=[sG.ep["weight"]])[:, -1].sum()
        global_weight = w_ranked[: sG.num_edges()].sum()
        if global_weight:
            if rich_weight > global_weight:
                pass
            results[rank] = rich_weight / global_weight
        else:
            results[rank] = np.nan
    return results


def _randomize_graph(G, Q=1, seed=None):
    R = G.copy()
    gt.random_rewire(R, n_iter=Q)
    rng = np.random.default_rng()
    rng.shuffle(R.ep["weight"].a)
    return _weighted_rich_club(R)


def weighted_rich_club(G, normalized=True, m=100, Q=100, seed=None, threads=None, alpha=0.05):
    result = _weighted_rich_club(G)
    degree = G.degree_property_map("total", weight=G.ep["weight"]).a
    k_values = np.unique(np.sort(degree))
    randoms = np.empty((m, len(k_values)))
    for i, _G in enumerate(it.repeat(G, m)):
        randoms[i] = _randomize_graph(_G)
    percentileofscore = np.vectorize(scs.percentileofscore, signature="(n),()->()")


    return pd.DataFrame(
        {
            "deviation": result,
            "k": k_values,
            "rand_deviation": np.mean(randoms, axis=0),
            "rand_deviation_std": np.std(randoms, axis=0),
            "rand_ci_lower": np.percentile(randoms, (alpha / 2) * 100, axis=0),
            "rand_ci_upper": np.percentile(randoms, (1 - (alpha / 2)) * 100, axis=0),
            "pvalue": percentileofscore(randoms.T, result),
        }
    )