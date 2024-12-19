import os
from pathlib import Path
import numpy as np

import faiss
import h5py

class FAISSIndex:
    """Indexer using FAISS with useful methods to handle embeddings

    Parameters
    ----------
    index_method : str
        The method used to index the embeddings. It can be "l2", "cosine", "IVF" or "PQ".
    d : int
        The dimension of the embeddings.
    normalize : bool
        Whether to normalize the embeddings before indexing.
        By default, it is set to False for all methods except for "cosine".
    """

    def __init__(self, index_method: str="l2", d: int=128, normalize: bool=False, index: faiss.Index=None):
        self.index_method = index_method
        self.normalize = normalize

        if index_method == "l2":
            self.index = faiss.IndexFlatL2(d)
        elif index_method == "cosine":
            self.normalize = True
            self.index = faiss.IndexFlatIP(d)  # Inner product
        elif index_method == "IVF":
            self.index = faiss.IndexHNSWFlat(d, 32)  # Hierarchical Navigable Small World
        elif index_method == "PQ":
            self.index = faiss.IndexPQ(d, 8, 8)  # Product quantization

        pass

    def add_h5_to_index(self, embeddings_path: str) -> None:
        """Create a faiss index from the embeddings
        NB: The index stores vectors with integers keys. In
        order to retrieve the id of the material, it is necessary to have a mapping
        between the id and the index of the vector in the embeddings file.

        Parameters
        ----------
        embeddings_path : str
            The path to the h5 file containing the embeddings.
        """
        file = h5py.File(embeddings_path, "r")

        for key in file.keys():
            x = file[key][()].reshape(1, -1)
            if self.normalize:
                faiss.normalize_L2(x)

            self.index.add(x)

        print(f"Number of embeddings: {self.index.ntotal}")

        file.close()

    def add_embeddings_path_to_index(self, embeddings_path: str) -> None:

        embeddings_paths = [
            Path(embeddings_path) / f
            for f in os.listdir(embeddings_path)
            if f.endswith(".h5")
        ]

        for embedding_path in embeddings_paths:
            self.add_h5_to_index(embedding_path)

    def add_dict_to_index(self, features_dict: dict[str, np.ndarray]) -> None:

        for key in features_dict:
            x = features_dict[key].reshape(1, -1)
            if self.normalize:
                faiss.normalize_L2(x)

            self.index.add(x)

    def add(self, x: np.ndarray, id: int=None) -> None:
        if self.normalize:
            faiss.normalize_L2(x)

        if id is not None:
            self.index.add_with_ids(x, id)
        else:
            self.index.add(x)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the k nearest neighbors of the query in the index
        """

        if self.normalize:
            faiss.normalize_L2(query)

        return self.index.search(query, k)

    def write_index(self, store_path: str) -> None:
        faiss.write_index(self.index, store_path)

    @staticmethod
    def from_store(store_path: str) -> "FAISSIndex":

        index = faiss.read_index(store_path)

        return FAISSIndex(index=index)


class FAISSKMeans:
    """KMeans clustering using FAISS

    Parameters
    ----------
    ncentroids : int
        Number of clusters
    niter : int
        Number of iterations
    verbose : bool
        Whether to print the progress
    enable_reverse : bool
        Whether to enable reverse search. This needs to fit the embeddings in memory by creating an index so will take longer and use more memory.
    """
    def __init__(self, ncentroids, niter=20, verbose=True, enable_reverse=False):
        self.ncentroids = ncentroids
        self.niter = niter
        self.verbose = verbose
        self.enable_reverse = enable_reverse

        self.is_trained = False

    def train(self, x):
        self.d = x.shape[1]
        self.kmeans = faiss.Kmeans(
            self.d, self.ncentroids, niter=self.niter, verbose=self.verbose
        )
        self.kmeans.train(x)

        if self.enable_reverse:
            self.reverse_index = faiss.IndexFlatL2(self.d)
            self.reverse_index.add(x)

        self.is_trained = True

    def _check_is_trained(self):
        assert self.is_trained, "KMeans is not trained"

    def search(self, query):
        self._check_is_trained()

        distances, centroids = self.kmeans.index.search(query, 1)
        return distances, centroids

    def reverse_search(self, query, topk=1):
        self._check_is_trained()

        assert self.enable_reverse, "Reverse search is not enabled"

        return self.reverse_index.search(query, topk)


class FAISSPCA:
    """Principal Component Analysis using FAISS

    Parameters
    ----------
    ncomp : int
        Number of components to reduce the PCA to.
    """
    def __init__(self, ncomp):
        self.ncomp = ncomp
        self.is_trained = False

    def _check_is_trained(self):
        assert self.is_trained, "PCA is not trained"

    def train(self, x):
        self.d = x.shape[1]
        self.pca = faiss.PCAMatrix(self.d, self.ncomp)
        self.pca.train(x)
        self.is_trained = True

    def apply(self, x):
        self._check_is_trained()
        return self.pca.apply(x)


if __name__ == "__main__":
    index = FAISSIndex.from_store("../crystals-query/index.faiss")
    from utils import concatenate_embeddings_dicts, embeddings_dict_to_numpy

    embeddings_dict = concatenate_embeddings_dicts("./data/eqv2_small/embeddings_test/")
    embeddings_x, ids = embeddings_dict_to_numpy(embeddings_dict)

    kmeans = FAISSKMeans(100)
    kmeans.train(embeddings_x)

    distances, clusters = kmeans.search(embeddings_x)

    import matplotlib.pyplot as plt
    import numpy as np

    pca = FAISSPCA(2)
    pca.train(embeddings_x)

    embeddings_x_pca = pca.apply(embeddings_x)

    plt.scatter(
        embeddings_x_pca[:, 0], embeddings_x_pca[:, 1], c=clusters, cmap="viridis", s=1
    )
    plt.savefig("pca.png")
    plt.show()

    import ipdb

    ipdb.set_trace()
