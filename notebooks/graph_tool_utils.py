import numpy as np
import pandas as pd
import graph_tool.all as gt


def graph_from_weighted_matrix(arr):
    arr = np.asarray(arr)
    es = np.nonzero(arr)
    g = gt.Graph(directed=False)
    g.add_vertex(arr.shape[0])
    g.add_edge_list(np.array(es).T)
    g.ep["weight"] = g.new_edge_property("double", vals=arr[es])
    return g


class VFilt:
    def __init__(self, g):
        self.g = g
    def __getitem__(self, index):
        ma = self.g.new_vp("bool", val=False)
        ma.a[index] = True
        return gt.GraphView(self.g, ma)



gt.Graph.vfilt = property(VFilt)

class VProps:
    def __init__(self, g):
        self.g = g

    def __getitem__(self, key):
        vp = self.g.vp[key]
        vp._g = self.g
        return vp

class EProps:
    def __init__(self, g):
        self.g = g

    def __getitem__(self, key):
        ep = self.g.ep[key]
        ep._g = self.g
        return ep

gt.Graph.v = property(VProps)
gt.Graph.e = property(EProps)

def _property_array(p, dtype=None):
    if "vector" in p.value_type():
        raise TypeError(
            f"Vector type property maps cannot be directly turned into arrays"
        )
    if "string" in p.value_type():
        return p.get_2d_array([0])[0]
    fa = p.fa
    if fa is None:
        getter = gt.Graph.get_edges if p.key_type() == "e" else gt.Graph.get_vertices
        return getter(p.get_graph(), [p])[:,-1]
    return np.array(fa)


gt.VertexPropertyMap.__array__ = _property_array
gt.EdgePropertyMap.__array__ = _property_array

def edge_index(g):
    ei = g.edge_index
    ei._g = g
    return ei

def vertex_index(g):
    ei = g.vertex_index
    ei._g = g
    return ei

gt.Graph.eindex = property(edge_index)

gt.Graph.vindex = property(vertex_index)



def get_eprops(g, eprop, vfilt=None):
    if vfilt is not None:
        g = gt.GraphView(g, vfilt=vfilt)
    return g.get_edges([eprop])[:, -1]


def _encode(feature):
    series = pd.Series(feature)
    return series.map(dict(map(reversed, enumerate(series.unique()))))


def encode_features(*features):
    if len(set(map(len, features))) > 1:
        raise ValueError("All features must be the same length")
    return np.r_[
        (
            "0,2",
            *(_encode(feature) for feature in features),
            np.zeros((len(features[0]),)),
        )
    ]


def partition_graph(g, partitions):
    if partitions.shape[1] != g.num_vertices():
        raise TypeError(
            "partitions must be a (p x n) array where n is equal to the number of "
            f"vertices in the graph: numvertices ({g.num_vertices()} != "
            f"{partitions.shape[1]})"
        )

    def split_partitions(base, parent):
        """Propogate parent partitions into child partitions"""
        base_range = np.max(base) - np.min(base) + 1
        for i, x in enumerate(np.unique(parent)):
            base[parent == x] += base_range * i

    def condense_partition(base, parent):
        """Reduce parent partition size to child

        Keep only enough entries in parent to cover the number of unique entries in the
        child
        """
        sort_ids = np.argsort(base)
        uniques = np.nonzero(np.r_[1, np.diff(base[sort_ids])[:-1]])
        return parent[sort_ids][uniques]

    # iterate over all partitions in pairs, starting with the last two and proceeding
    # backward
    partitions = np.copy(partitions)
    for i, part in enumerate(partitions[1:][::-1]):
        split_partitions(partitions[-2 - i], part)

    # iterate over partitions in pairs starting with first two
    layers = [partitions[0]]
    for i, part in enumerate(partitions[1:]):
        layers.append(condense_partition(partitions[i], part))

    return gt.NestedBlockState(g, layers)


def rotate_layout(layout, radians):
    x, y = gt.ungroup_vector_property(layout, [0, 1])
    x_off = x.fa.mean()
    y_off = y.fa.mean()
    x.fa -= x_off
    y.fa -= y_off
    mag = np.sqrt(x.fa**2 + y.fa**2)
    angle = np.arctan2(y.fa, x.fa) - radians
    x.fa = np.cos(angle) * mag
    y.fa = np.sin(angle) * mag
    x.fa += x_off
    y.fa += y_off
    return gt.group_vector_property([x, y])
