import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import tqdm
import yaml
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
import ase

from fairchem.core import OCPCalculator
from fairchem.core.common.tutorial_utils import generate_yml_config

from material_hasher.similarity.utils import get_atoms_from_row


class BaseFairChemEmbedder:
    """ABClass to embed structures using a FairChem model. 
    """

    def add_model_hook(self):
        raise NotImplementedError

    def load_model_from_path(self, model_path, cpu=False):
        calc = OCPCalculator(checkpoint_path=model_path, cpu=cpu)
        if not self.trained:
            config = calc.trainer.config

            config["dataset"] = {
                "train": {"src": "dummy"}
            }  # for compatibility with yaml loading

            yaml.dump(config, open("/tmp/config.yaml", "w"))
            calc = OCPCalculator(config_yml="/tmp/config.yaml", cpu=cpu)

        self.calc = calc
        self.add_model_hook()
        return calc


class FairChemEmbedder(BaseFairChemEmbedder):
    """The models is loaded using OCPCalculator which means that batched inference is not possible with this class.
    Only tested with EquiformerV2 models for now.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint
    trained : bool
        Whether the model was trained or not
    device : str
        Device to use for inference (e.g. "cuda:0" or "cpu")
    """

    def __init__(self, model_path: str, trained: bool, device: str):
        self.model_path = model_path
        self.trained = trained
        self.device = device

        self.calc = None
        self.features = {}

    def add_model_hook(self):
        assert self.calc is not None, "Model not loaded"

        def hook_norm_block(m, input_embeddings, output_embeddings):
            self.features["sum_norm_embeddings"] = (
                output_embeddings.narrow(1, 0, 1)
                .squeeze(1)
                .sum(0)
                .detach()
                .cpu()
                .numpy()
            )

        self.calc.trainer.model.backbone.norm.register_forward_hook(hook_norm_block)

    def relax_atoms(self, atoms: ase.Atoms) -> ase.Atoms:
        """ Relax the atoms using the FIRE optimizer
        TODO: Add the option to use the BFGS optimizer (or other)
        WARNING: This function modifies the atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object to relax
        """
        atoms.calc = self.calc

        dyn = FIRE(FrechetCellFilter(atoms))
        dyn.run(steps=0)

        return atoms

    def compute_embeddings(self, dataset, embeddings_path):
        file = h5py.File(embeddings_path, "a")

        for i, row in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            if str(i) in file.keys():
                continue

            atoms = get_atoms_from_row(row)
            atoms = self.relax_atoms(atoms)

            # there are no unique keys in the dataset for now (immutable_id is not unique), use the row index...
            file.create_dataset(
                str(i),
                data=self.features["sum_norm_embeddings"],
            )

            print(f"Energy (Equiformer): {atoms.get_potential_energy()}")
            print(f"Energy (Dataset): {row['energy']}")

    def get_structure_embeddings(self, structure):
        atoms = get_atoms_from_row(structure)
        atoms = self.relax_atoms(atoms)

        return self.features["sum_norm_embeddings"]


class BatchedFairChemEmbedder(BaseFairChemEmbedder):
    """FairChemEmbedder with batched inference capabilities.
    This requires having created existing .aselmdb databases to run inference on.

    Parameters
    ----------
    model_path : str
        Path to the model checkpoint
    trained : bool
        Whether the model was trained or not
    device : str
        Device to use for inference (e.g. "cuda:0" or "cpu")
    batch_size : int
        Batch size to use for inference
    buffer_size : int
        Number of embeddings to save in the buffer before saving them to an h5 file
    """


    def __init__(
        self,
        model_path,
        trained,
        device,
        batch_size,
        buffer_size=1000,
    ):
        self.model_path = model_path
        self.trained = trained
        self.device = device
        self.batch_size = batch_size

        self.calc = None
        self.features = {}
        self.h5_file = None
        self.mapping_dict = None
        self.buffer_size = buffer_size

    def save_features_h5(self):
        for k, v in self.features.items():
            self.h5_file.create_dataset(str(k), data=v)  # Keys do not support int
        self.features.clear()

    def add_model_hook(self):
        """
        Add a hook to the model to save the embeddings of the input batch to a h5 file.
        The embeddings are extracted at the entry of the energy output block.
        Batches are split using their number of atoms and the
        key of the embeddings are their position in the .aselmdb database.
        To avoid saving too many embeddings at once, a buffer is used to save the embeddings in batches to a h5 file. This can be disabled by setting the h5_file to None.
        """
        assert self.calc is not None, "Model not loaded"

        def hook_energy_block(m, input_embeddings, output_embeddings):
            input_graph = input_embeddings[0]

            natoms = input_graph.natoms.cpu().numpy()
            sids = input_graph.sid

            embeddings = (
                input_embeddings[1]["node_embedding"]
                .embedding[:, 0, :]
                .detach()
                .cpu()
                .numpy()
            )

            indices = np.cumsum(natoms)[:-1]

            split_embeddings = np.split(embeddings, indices)

            if self.h5_file is not None:
                if len(self.features) > self.buffer_size:
                    self.save_features_h5()

            for sid, emb in zip(sids, split_embeddings):
                sid_ = str(int(sid.cpu()))
                if self.mapping_dict is not None:
                    sid_ = self.mapping_dict[sid_]
                self.features[sid_] = emb.sum(axis=0)

        self.calc.trainer.model.output_heads.energy.register_forward_hook(
            hook_energy_block
        )

        return self.calc, self.features

    def run_batched_inference_lmdb(self, db_path, h5_file_path=None, mapping_dict=None):
        self.mapping_dict = mapping_dict

        yml_path = generate_yml_config(
            self.model_path,
            "/tmp/config.yml",
            delete=[
                "logger",
                "task",
                "model_attributes",
                "dataset",
                "slurm",
                "optim.load_balancing",
            ],
            # Load balancing works only if a metadata.npz file is generated using the make_lmdb script (see: https://github.com/FAIR-Chem/fairchem/issues/876)
            update={
                "amp": True,
                "gpus": 1,
                "task.prediction_dtype": "float16",
                "logger": "tensorboard",  # don't use wandb!
                # Test data - prediction only so no regression
                "test_dataset.src": db_path,
                "test_dataset.format": "ase_db",
                "test_dataset.2g_args.r_energy": False,
                "test_dataset.a2g_args.r_forces": False,
                "optim.eval_batch_size": self.batch_size,
            },
        )

        config = yaml.safe_load(open(yml_path))
        config["dataset"] = {}
        config["val_dataset"] = {}

        self.calc.trainer.config = config
        self.calc.trainer.load_datasets()
        self.calc.trainer.is_debug = False

        if h5_file_path is not None:
            self.h5_file = h5py.File(h5_file_path, "w")

        self.calc.trainer.predict(
            self.calc.trainer.test_loader, self.calc.trainer.test_sampler
        )

        if h5_file_path is not None:
            self.save_features_h5()
            self.h5_file.close()

        return h5_file_path, self.features

    def run_batched_inference(
        self, dbs_path, save_embeddings_path, index_start=0, index_end=None
    ):
        os.makedirs(save_embeddings_path, exist_ok=True)

        files = os.listdir(dbs_path)
        dbs_paths = [
            os.path.join(dbs_path, file) for file in files if file.endswith(".aselmdb")
        ]
        dbs_paths.sort()
        mapping_paths = [
            os.path.join(dbs_path, file) for file in files if file.endswith(".json")
        ]
        mapping_paths = mapping_paths.sort()

        for i, (db_path, mapping_path) in enumerate(zip(dbs_paths, mapping_paths)):
            if (
                f"features_{i}.pkl" in os.listdir(save_embeddings_path)
                or i < index_start
                or (index_end is not None and i >= index_end)
            ):
                continue
            mapping_dict = json.load(open(mapping_path, "r"))
            _, features = self.run_batched_inference_lmdb(db_path, None, mapping_dict)
            pickle.dump(
                features, open(Path(save_embeddings_path) / f"features_{i}.pkl", "wb")
            )
            self.features.clear()
