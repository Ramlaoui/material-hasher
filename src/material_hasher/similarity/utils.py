import json
import pickle
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple

import numpy as np
import ase
import tqdm
from ase.calculators.singlepoint import SinglePointCalculator  # To add targets
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import datasets


def get_dataset_lemat_bulk() -> datasets.Dataset:
    datasets = []
    subsets = [
        "compatible_pbe",
        "compatible_pbesol",
        "compatible_scan",
        "non_compatible",
    ]

    for subset in subsets:
        dataset = load_dataset(
            "LeMaterial/leMat-Bulk",
            subset,
            columns=[
                "lattice_vectors",
                "species_at_sites",
                "cartesian_site_positions",
                "energy",
                # "energy_corrected", # not yet available in LeMat-Bulk
                "immutable_id",
                "elements",
                "functional",
                "stress_tensor",
                "magnetic_moments",
                "forces",
                # "band_gap_direct", #future release
                # "band_gap_indirect", #future release
                "dos_ef",
                # "charges", #future release
                "functional",
                "chemical_formula_reduced",
                "chemical_formula_descriptive",
                "total_magnetization",
                "entalpic_fingerprint",
            ],
        )
        datasets.append(dataset["train"])

    return concatenate_datasets(datasets)


def get_structure_from_row(row: Union[Dict, List]) -> Structure:
    structure = Structure(
        [x for y in row["lattice_vectors"] for x in y],
        species=row["species_at_sites"],
        coords=row["cartesian_site_positions"],
        coords_are_cartesian=True,
    )

    return structure


def get_atoms_from_row(row: Union[Dict, List], add_targets: bool=False, add_forces: bool=True) -> ase.Atoms:
    if isinstance(row, dict):
        # Convert row to PyMatGen
        structure = Structure(
            [x for y in row["lattice_vectors"] for x in y],
            species=row["species_at_sites"],
            coords=row["cartesian_site_positions"],
            coords_are_cartesian=True,
        )
    else:
        structure = row

    atoms = AseAtomsAdaptor.get_atoms(structure)

    # Add the forces and energy as targets
    if add_targets:
        if (
            add_forces
            and np.array(row["forces"]).shape[0]
            == np.array(row["cartesian_site_positions"]).shape[0]
        ):
            atoms.calc = SinglePointCalculator(
                atoms, forces=row["forces"], energy=row["energy"]
            )
        else:
            return None

    return atoms


def get_similarity_matrix(embeddings: np.ndarray, metric: str="cosine") -> np.ndarray:
    if metric == "cosine":
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(embeddings, embeddings)
    else:
        raise NotImplementedError

    return sim_matrix


def download_model(hf_repo_id: str, hf_model_path: str) -> str:
    model_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_model_path)
    return model_path


def subsample_dataset(dataset: datasets.Dataset, x_dataset: Optional[float]=None, seed: int=0, change_order: bool=False) -> datasets.Dataset:
    if x_dataset is not None:
        n_samples = int(np.floor(len(dataset) * x_dataset))
        if change_order:
            np.random.seed(seed)
            samples = np.random.choice(len(dataset), n_samples, replace=False)
        else:
            samples = np.arange(n_samples)
        dataset = dataset.select(samples)

    return dataset


def concatenate_embeddings_dicts(embeddings_path: str, output_path: Optional[str]=None) -> Dict[str, np.ndarray]:
    path = Path(embeddings_path)

    embeddings_dict = {}
    for feature_path in path.glob("*.pkl"):
        feature = pickle.load(open(feature_path, "rb"))
        embeddings_dict = {**embeddings_dict, **feature}

    if output_path is not None:
        output_path = Path(f"{str(output_path).replace('.pkl', '')}.pkl")
        pickle.dump(embeddings_dict, open(output_path, "wb"))

    return embeddings_dict


def embeddings_dict_to_numpy(embeddings_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    all_features = []
    keys = list(embeddings_dict.keys())
    for key in embeddings_dict:
        all_features.append(embeddings_dict[key].reshape(1, -1))

    return np.concatenate(all_features), keys
