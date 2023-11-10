import os
import re
import tempfile
import functools as ft

from bids.layout import parse_file_entities
from snakebids import bids, generate_inputs, filter_list
from snakemake import utils as sutils
from snakemake.exceptions import WorkflowError, IncompleteCheckpointException
from snakemake.io import checkpoint_target
import pandas as pd

from pathlib import Path
from snakeboost import Pyscript, PipEnv, Boost
from templateflow import api as tflow



tmpdir = eval(
    workflow.default_resources._args.get("tmpdir"),
    {"system_tmpdir": tempfile.gettempdir()}
)

if workflow.run_local:
    workflow.shadow_prefix = os.environ.get("SLURM_TMPDIR")

###
# Input Globals
###
dataset = config["dataset"]
inputs = generate_inputs(
    bids_dir=config['bids_dir'][dataset],
    pybids_inputs=config['pybids_inputs'],
    derivatives=True,
    participant_label=config.get("participant_label"),
    exclude_participant_label=config.get("exclude_participant_label"),
    pybids_database_dir=os.path.join(
        config['bids_dir'][dataset],
        config.get("pybids_database_dir")
    ),
    pybids_reset_database=config.get("pybids_reset_database"),
)


###
# Output Globals
###

work = Path(tmpdir) / "sn_prepdwi_recon"
shared_work = Path(config['output_dir'])/'work'/'prepdwi_recon'
output = Path(config['output_dir'])/"prepdwi_recon"
source = output/"sourcedata"
qc = Path(output)/"qc"


def shell_uid(sample):
    return '.'.join(
        re.sub(r'^\{', '{wildcards.', val)
        for val in inputs.input_wildcards[sample].values()
    )


###
# Utility functions
###
boost = Boost(work, logger, debug=True)
pyscript = Pyscript(workflow.basedir)

nodal_props_venv = PipEnv(
    packages = [
        "numpy",
        "pandas",
        "networkx",
        "colour",
        "/scratch/knavynde/snakeboost",
        "plotly",
        "attrs",
    ],
    flags = config.get("pip-flags", ""),
    root = work,
)
