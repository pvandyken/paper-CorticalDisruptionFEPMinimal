import pandas as pd

# %%
def read_metadata(file):
    """Turns the Brainnetome data table into something more useful
    
    Original data scraped from https://doi.org/10.1093/cercor/bhw157
    """
    raw = pd.read_csv(file)
    split_name_cols = (
        raw.assign(
            **dict(
                raw["Modified cyto-architectonic"]
                .str.split(r',\s?', expand=True)
                .rename({0: "Name", 1: "Long Name"}, axis=1)
                .drop(columns=[2])
            ),
            **dict(
                raw["Gyrus"]
                .str.split(r',\s?', expand=True)
                .rename({0: "Gyrus Abbr", 1: "Gyrus"}, axis=1)
            )
        )
        .drop(
            columns=[
                "Modified cyto-architectonic",
                "Left and right hemispheres"
            ]
        )
    )

    unstack_hemispheres = (
        pd.concat([
            split_name_cols
            .drop(columns=["rh.MNI (X, Y, Z)", "Label ID.R"])
            .rename(columns={"lh.MNI (X,Y,Z)": "MNI", "Label ID.L": "Label ID"})
            .assign(hemisphere="L"),

            split_name_cols
            .drop(columns=["lh.MNI (X,Y,Z)", "Label ID.L"])
            .rename(columns={"rh.MNI (X, Y, Z)": "MNI", "Label ID.R": "Label ID"})
            .assign(hemisphere="R")
        ])
        .astype({"Label ID": int})
        .assign(
            Lobe=lambda df: df["Lobe"].str.strip(),
            MNI=lambda df: df["MNI"].str.strip(),
        )
        .set_index("Label ID")
        .sort_index()
    )

    unstack_hemispheres["MNI"] = unstack_hemispheres["MNI"].str.replace("−", "-")

    return (
        unstack_hemispheres
        .assign(
            **dict(
                unstack_hemispheres["MNI"]
                .str.split(r',\s?', expand=True)
                .rename(columns={0: "MNI X", 1: "MNI Y", 2: "MNI Z"})
            )
        )
    )

