import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tqdm
from ase import Atoms

from fairchem.core.datasets.lmdb_database import LMDBDatabase
from material_hasher.similarity.utils import get_atoms_from_row, get_dataset_lemat_bulk, subsample_dataset


def create_ase_db(
    dataset, output_path:str, separate_db: Optional[int]=None, add_targets: bool=False, add_forces: bool=True
):
    """Create an ASE DB from a HF dataset (works for LeMaterial)

    Parameters
    ----------
    dataset : Dataset
        The Hugging Face dataset to create the ASE DB from
    output_path : str
        The path to the output ASE DB
    separate_db : Optional[int], optional
        Number of separate DBs to create, by default None
    add_targets : bool, optional
        Add targets to the ASE atoms object (energy and forces), by default False
    add_forces : bool, optional
        Add forces to the ASE atoms object (some rows are missing forces, this will skip them), by default True
    """

    # REF: https://github.com/FAIR-Chem/fairchem/issues/787
    assert output_path.endswith(".aselmdb"), "Output path must end with .aselmdb"

    # Currently, there are no unique ids in LeMat dataset, using immutable_id, functional

    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    db_length = len(dataset) // separate_db if separate_db is not None else len(dataset)

    assert (db_length + 1) * separate_db >= len(dataset), "Separate DBs are too small"

    for i in range(separate_db):
        mappings = {}
        range_start = i * db_length
        range_end = min((i + 1) * db_length, len(dataset))
        for j, row in tqdm.tqdm(
            enumerate(dataset.select(range(range_start, range_end))),
            total=range_end - range_start,
        ):
            with LMDBDatabase(
                f"{str(output_path).split('.aselmdb')[0]}_{i}.aselmdb"
            ) as db:
                atoms = get_atoms_from_row(
                    row, add_targets=add_targets, add_forces=add_forces
                )

                mappings[j] = str(row["immutable_id"]) + "_" + str(row["functional"]) # immutable_id is sometimes None
                db.write(atoms, data={"id": row["immutable_id"] + "_" + row["functional"]})
        json.dump(mappings, open(output_path.parent / f"mapping_{i}.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/lemat_bulk_lmdb/lemat_bulk.aselmdb",
        help="Path to the output ASE DB",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1,
        help="Subsample the dataset by a factor",
    )
    parser.add_argument(
        "--separate-db",
        type=int,
        default=None,
        help="Number of separate DBs to create",
    )
    parser.add_argument(
        "--add-targets",
        action="store_true",
        help="Add targets to the ASE atoms object (energy and forces)",
    )
    parser.add_argument(
        "--add-forces",
        action="store_true",
        help="Add forces to the ASE atoms object (some rows are missing forces, this will skip them)",
    )

    args = parser.parse_args()

    dataset = get_dataset_lemat_bulk()
    dataset = subsample_dataset(dataset, args.subsample, change_order=True)

    create_ase_db(
        dataset, args.output_path, args.separate_db, args.add_targets, args.add_forces
    )
