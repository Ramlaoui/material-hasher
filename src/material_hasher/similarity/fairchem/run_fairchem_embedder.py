import argparse

from material_hasher.similarity.fairchem.embedder import BatchedFairChemEmbedder, FairChemEmbedder
from material_hasher.similarity.utils import download_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    help = """
    This script is used to embed a dataset of materials into a fixed-size vector representation using EquiformerV2.
    The embeddings are generated using a pretrained model from the Hugging Face model hub.
    """

    parser.add_argument(
        "--hf_model_repo_id",
        type=str,
        default="fairchem/OMAT24",
        help="Hugging Face model repository ID",
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default="eqV2_31M_omat_mp_salex.pt",
        help="Hugging Face model path",
    )
    parser.add_argument(
        "--trained", type=bool, default=True, help="Whether the model is trained"
    )
    parser.add_argument("--cpu", type=bool, default=False, help="Whether to use CPU")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--lmdb_dataset_path",
        type=str,
        default="data/lmdb",
        help="Path to the LMDB datasets",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/embeddings_test",
        help="Path to save the embeddings",
    )
    parser.add_argument(
        "--index-start",
        type=int,
        default=0,
        help="DB index to start embeddings from",
    )
    parser.add_argument(
        "--index-end",
        type=int,
        default=None,
        help="DB index to end embeddings from. Can be useful to parallelize the embeddings generation",
    )

    args = parser.parse_args()

    model_path = download_model(args.hf_model_repo_id, args.hf_model_path)

    batched_embedder = BatchedFairChemEmbedder(
        model_path, args.trained, args.cpu, args.batch_size
    )
    batched_embedder.load_model_from_path(model_path, args.cpu)
    batched_embedder.run_batched_inference(
        args.lmdb_dataset_path, args.output_path, args.index_start, args.index_end
    )
