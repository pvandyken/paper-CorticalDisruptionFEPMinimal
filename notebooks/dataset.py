from collections import defaultdict
import pandas as pd
import numpy as np
import more_itertools as itx
from bids import BIDSLayout
import copy
from pathlib import Path
import functools as ft
import itertools as it

from notebooks.adjacency_matrix import AdjacencyMatrix
import xarray as xr

cats = {
    "chronic": "chronic",
    "CHR": "High risk",
    "HC": "HC",
    "FEP": "FEP",
    "Control": "HC",
    "Patient": "Patient",
    "Schizophrenia_Strict": "Patient",
    "No_Known_Disorder": "HC",
    "Schizoaffective": "Schizoaffective",
}


def set_participant_index(df):
    def get_sid(s):
        try:
            return int(s[4:])
        except ValueError:
            return s[4:]

    return df.assign(subject=df["participant_id"].map(get_sid)).set_index("subject")


def _set_subsess_index(df):
    index = [df["participant_id"].map(lambda s: s[4:])]
    drop = ["participant_id"]
    if "session_id" in df:
        index.append(df["session_id"].map(lambda s: s[4:]))
        drop.append("session_id")
    return df.set_index(index, drop=True).drop(columns=drop)


def get_subj_metadata(layout):
    base = itx.one(
        layout.get(scope="raw", suffix="participants", extension=".tsv")
    ).path

    # deriv = pd.read_csv(
    #     itx.one(
    #         layout.get(
    #             scope=["prepdwi_recon", "snaketract"],
    #             suffix="participants",
    #             extension=".tsv"
    #         )
    #     ).path,
    #     sep="\t",
    # )
    def get_group(df):
        if "phenotype" in df:
            group = df["phenotype"]
        else:
            group = df["dx"]
        return group.map(cats)

    return (
        pd.read_csv(base, sep="\t").pipe(_set_subsess_index)
        # .loc[lambda df: df["participant_id"].isin(deriv["participant_id"])]
        .assign(group=get_group)
    )


def get_layout(dataset):
    dataset = Path(dataset)
    return BIDSLayout(
        dataset, derivatives=True, database_path=dataset / ".pybids", validate=False
    )


class Atlas:
    bn246 = pd.read_csv("resources/atlases/atlas-brainnetome246/labels.tsv", sep="\t")


class Bn246:
    def __init__(self):
        self.metadata = pd.read_csv(
            "resources/atlases/atlas-brainnetome246/labels.tsv", sep="\t"
        )

    def create_edgelist(self, arr):
        edges = np.triu_indices_from(arr, k=1)
        weight = arr[edges]
        mask = np.zeros_like(weight, dtype=np.bool_)
        mask[np.nonzero(weight)] = True
        return {"edge": ("edge", weight), "edgemask": ("edge", mask)}


def dict_defaults(__dict, **defaults):
    return {**defaults, **__dict}


class Dataset:
    def __init__(self, path, label):
        self.path = Path(path)
        self.label = label
        layout = get_layout(self.path)
        self.layout = layout
        self.metadata = get_subj_metadata(layout)
        self._exclude_subj = set()

    def __getitem__(self, item):
        new = copy.copy(self)
        new.metadata = self.metadata[item]
        return new

    def exclude_subjects(self, subjects):
        self._exclude_subj.update(subjects)
        return self

    @property
    def subjects(self):
        return (
            set(self.metadata.index.unique("participant_id")) - self._exclude_subj
        )


    def get(self, **entities):
        subjects = self.subjects

        if (
            entities.get("subject") not in subjects
            and entities.get("subject") is not None
        ):
            return []
        return self.layout.get(
            **dict_defaults(entities, subject=list(subjects), scope="snaketract")
        )

    def newadj(self, **entities):
        adjs = []

        entries = self.get(
            **dict_defaults(entities, suffix="connectome", atlas="bn246")
        )
        if not entries:
            return []

        entities = entries[0].get_entities()
        baseshape = tuple(len(dim) for dim in entities.values())
        atlas = Bn246()
        entries_ = map(
            lambda entry: atlas.create_edgelist(
                np.genfromtxt(entry.path, delimeter=",")
            )
            | entry.get_entities(),
            entries,
        )
        first = itx.first(entries_)
        edges = np.full((first["edge"].shape[0], *baseshape), np.nan)
        mask = np.full((first["edgemask"], *baseshape), np.nan)
        for data in it.chain([first], entries_):
            edges
            try:
                adjs.append(
                    AdjacencyMatrix(
                        raw=np.genfromtxt(entry.path, delimiter=",")[1:, 1:],
                        metadata=Atlas.bn246,
                        attrs=entry.get_entities(),
                    )
                    .mask_diagonal()
                    .mask_equal(0)
                    # .mask_where_meta(MetadataMasks.src_and_dest("hemisphere").equals("L"))
                )
            except KeyError as err:
                print(self.metadata)
                raise err
            adjs[-1].props["distance"] = np.ma.filled(1 / adjs[-1].raw, np.NaN)
        return adjs

    def adj(self, **entities):
        adjs = []

        for entry in self.get(
            **dict_defaults(entities, suffix="connectome", atlas="bn246")
        ):
            if Path(entry.path).suffix != ".csv":
                raise TypeError("adj files must be csv files")
            try:
                adjs.append(
                    AdjacencyMatrix(
                        raw=np.genfromtxt(entry.path, delimiter=",")[1:, 1:],
                        metadata=Atlas.bn246,
                        attrs=entry.get_entities(),
                    )
                    .mask_diagonal()
                    .mask_equal(0)
                    # .mask_where_meta(MetadataMasks.src_and_dest("hemisphere").equals("L"))
                )
            except KeyError as err:
                print(self.metadata)
                raise err
            adjs[-1].props["distance"] = np.ma.filled(1 / adjs[-1].raw, np.NaN)
        return adjs

    def merged_metadata(self, atlas):
        return xr.merge(
            [
                self.metadata.to_xarray(),
                atlas.reset_index()
                .rename(columns={"Label ID": "node"})
                .set_index("node")
                .to_xarray(),
            ]
        )

    def nodal_props(self, **entities):
        data = (
            pd.read_csv(
                f"results/{self.label}_nodes.tsv",
                sep="\t",
                dtype={"subject": str, "session": str},
            )
            .rename(columns={"subject": "participant_id", "session": "session_id"})
            .assign(node=lambda df: df["node"] + 1)
        )
        index_cols = set(data.columns) - {
            "degree",
            "clust_coeff",
            "path_length",
            "betweenness",
            "category",
        }
        data = (
            data.set_index(list(index_cols))
            .to_xarray()
            .drop_sel(
                participant_id=list(
                    set(data["participant_id"]) & set(self._exclude_subj)
                )
            )
        )
        return self.merged_metadata(Atlas.bn246).merge(data, join="inner")

    def add_phenotypes(
        self,
        spec,
    ):
        phen_path = Path(self.layout.root, "phenotypes")
        by_file = defaultdict(dict)
        for field, loc in spec.items():
            by_file[loc["file"]][field] = loc["field"]
        dfs = []
        for stem, field_map in by_file.items():
            file = phen_path / Path(stem).with_suffix(".tsv")
            rename_map = dict(zip(field_map.values(), field_map))
            dfs.append(
                pd.read_csv(file, sep="\t")
                .pipe(_set_subsess_index)[list(rename_map)]
                .rename(columns=rename_map)
                .pipe(
                    lambda df: df[df["session_id"] == "1"] if "session_id" in df else df
                )
            )
        if not dfs:
            return self
        result = copy.copy(self)
        metadata = ft.reduce(
            lambda df1, df2: df1.join(df2, how="outer"), dfs, self.metadata
        )
        for field, attrs in spec.items():
            if "map" in attrs:
                metadata[field] = metadata[field].map(attrs["map"])
            if "valid" in attrs:
                metadata[field] = metadata[field].where(
                    metadata[field].isin(attrs["valid"])
                )
            if "invalid" in attrs:
                metadata[field] = metadata[field].where(
                    ~metadata[field].isin(attrs["invalid"])
                )

        result.metadata = metadata
        return result
