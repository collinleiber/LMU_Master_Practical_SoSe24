import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from enum import Enum

import torch
from clustpy.data import load_fmnist, load_mnist, load_reuters, load_usps
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep.dec import IDEC
from clustpy.metrics import unsupervised_clustering_accuracy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import Bunch

from practical.DeepClustering.DeepECT.deepect import DeepECT


class DatasetType(Enum):
    MNIST = "MNIST"
    FASHION_MNIST = "Fashion MNIST"
    USPS = "USPS"
    REUTERS = "Reuters"


class FlatClusteringMethod(Enum):
    DEEPECT = "DeepECT"
    DEEPECT_AUGMENTED = "DeepECT + Augmentation"
    IDEC = "IDEC"
    KMEANS = "KMeans"


class HierarchicalClusteringMethod(Enum):
    DEEPECT = "DeepECT"
    DEEPECT_AUGMENTED = "DeepECT + Augmentation"
    IDEC_SINGLE = "IDEC + Single"
    IDEC_COMPLETE = "IDEC + Complete"
    AE_BISECTING = "Autoencoder + Bisection"
    AE_SINGLE = "Autoencoder + Single"
    AE_COMPLETE = "Autoencoder + Complete"


def calculate_nmi(true_labels, predicted_labels):
    """
    Calculate the Normalized Mutual Information (NMI) between true and predicted labels.

    Parameters
    ----------
    true_labels : array-like
        The true labels of the data points.
    predicted_labels : array-like
        The predicted labels of the data points.

    Returns
    -------
    nmi : float
        The NMI score.
    """
    return normalized_mutual_info_score(true_labels, predicted_labels)


def calculate_acc(true_labels, predicted_labels):
    """
    Calculate the Clustering Accuracy (ACC) between true and predicted labels.

    Parameters
    ----------
    true_labels : array-like
        The true labels of the data points.
    predicted_labels : array-like
        The predicted labels of the data points.

    Returns
    -------
    acc : float
        The accuracy score.
    """
    return unsupervised_clustering_accuracy(true_labels, predicted_labels)


def calculate_ari(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)


def get_max_epoch_size(data, max_iterations, batch_size):
    return math.ceil(max_iterations / (len(data) / batch_size))


def pretraining(
    init_autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset: Bunch,
    seed: int,
):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load and preprocess data
    data = dataset["data"]

    # TODO: Check for augmentation here

    # Initialize the autoencoder
    autoencoder: _AbstractAutoencoder = init_autoencoder(
        layers=[data.shape[1], 500, 500, 2000, 10], reusable=True
    )

    if not os.path.exists(autoencoder_params_path):
        # Train the autoencoder if parameters file does not exist
        autoencoder.to(device)
        autoencoder.fit(
            n_epochs=get_max_epoch_size(data, 130000, 256),
            optimizer_params={"lr": 0.0001},
            data=data,
            batch_size=256,
            device=device,
        )
        autoencoder.save_parameters(autoencoder_params_path)
        print("Autoencoder pretraining complete and saved.")
    else:
        # Load the existing parameters
        autoencoder.load_parameters(autoencoder_params_path)
        print("Autoencoder parameters loaded from file.")

    return autoencoder


def flat(
    autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset_type: DatasetType,
    dataset: Bunch,
    seed: int,
):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load and preprocess data
    data = dataset["data"]
    labels = dataset["target"]
    results = []
    max_iterations = 50000
    batch_size = 256
    max_leaf_nodes = 12 if dataset_type == DatasetType.REUTERS else 20
    n_clusters = 4 if dataset_type == DatasetType.REUTERS else 10

    max_clustering_epochs = get_max_epoch_size(data, max_iterations, batch_size)

    for method in FlatClusteringMethod:
        # Load the autoencoder parameters
        autoencoder.load_parameters(autoencoder_params_path)
        autoencoder.fitted = True

        if method == FlatClusteringMethod.KMEANS:
            autoencoder.to(device)
            # Encode the data
            embeddings = (
                autoencoder.encode(
                    torch.tensor(data, dtype=torch.float32, device=device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            # Perform flat clustering with KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                init="random",
                n_init=20,
                random_state=seed,
                max_iter=max_iterations,
            )
            print("fitting KMeans...")
            predicted_labels = kmeans.fit_predict(embeddings)
            print("finished fitting Kmeans")

            # Calculate evaluation metrics
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "nmi": calculate_nmi(labels, predicted_labels),
                    "acc": calculate_acc(labels, predicted_labels),
                    "ari": calculate_ari(labels, predicted_labels),
                    "seed": seed,
                }
            )
        elif method == FlatClusteringMethod.DEEPECT:
            autoencoder.to(device)
            deepect = DeepECT(
                autoencoder=autoencoder,
                clustering_optimizer_params={"lr": 1e-4, "betas": (0.9, 0.999)},
                max_leaf_nodes=max_leaf_nodes,
                random_state=np.random.RandomState(seed),
            )
            deepect.fit(data)
            # Calculate evaluation metrics
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "nmi": deepect.tree_.flat_nmi(labels, n_clusters),
                    "acc": deepect.tree_.flat_accuracy(labels, n_clusters),
                    "ari": deepect.tree_.flat_ari(labels, n_clusters),
                    "dp": deepect.tree_.dendrogram_purity(labels),
                    "lp": deepect.tree_.leaf_purity(labels)[0],
                    "seed": seed,
                }
            )

        elif method == FlatClusteringMethod.DEEPECT_AUGMENTED:
            # Perform flat clustering with DeepECT and augmentation
            if dataset_type == DatasetType.REUTERS:
                pass

            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "nmi": 0,
                    "acc": 0,
                    "ari": 0,
                    "seed": seed,
                }
            )

        elif method == FlatClusteringMethod.IDEC:
            # Perform flat clustering with IDEC
            idec = IDEC(
                n_clusters=n_clusters,
                batch_size=batch_size,
                autoencoder=autoencoder,
                clustering_epochs=max_clustering_epochs,
                random_state=seed,
                initial_clustering_class=KMeans,
                initial_clustering_params={
                    "init": "random",
                    "n_init": 20,
                    "random_state": seed,
                },
            )
            print("fitting IDEC...")
            idec.fit(data)
            print("finished fitting IDEC")

            predicted_labels = idec.predict(data)
            # Calculate evaluation metrics
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "nmi": calculate_nmi(labels, predicted_labels),
                    "acc": calculate_acc(labels, predicted_labels),
                    "ari": calculate_ari(labels, predicted_labels),
                    "seed": seed,
                }
            )

    df_results = pd.DataFrame(results)
    return df_results


def hierarchical(
    autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset_type: DatasetType,
    dataset: Bunch,
    seed: int,
):
    # Set the seed for reproducibility
    torch.manual_seed(seed)

    # Load and preprocess data
    data = dataset["data"]
    labels = dataset["target"]
    results = []
    max_leaf_nodes = 12 if dataset_type == DatasetType.REUTERS else 20
    n_clusters = 4 if dataset_type == DatasetType.REUTERS else 10

    for method in HierarchicalClusteringMethod:
        # Load the autoencoder parameters
        autoencoder.load_parameters(autoencoder_params_path)
        autoencoder.fitted = True
        if method == HierarchicalClusteringMethod.IDEC_SINGLE:
            # Perform hierarchical clustering with IDEC and single
            pass
        elif method == HierarchicalClusteringMethod.IDEC_COMPLETE:
            # Perform hierarchical clustering with IDEC and complete
            pass
        elif method == HierarchicalClusteringMethod.AE_BISECTING:
            # Perform hierarchical clustering with Autoencoder and bisection
            pass
        elif method == HierarchicalClusteringMethod.AE_SINGLE:
            # Perform hierarchical clustering with Autoencoder and single
            pass
        elif method == HierarchicalClusteringMethod.AE_COMPLETE:
            # Perform hierarchical clustering with Autoencoder and complete
            pass

    df_results = pd.DataFrame(results)
    return df_results


# Example usage
def evaluate(
    init_autoencoder: _AbstractAutoencoder,
    dataset_type: DatasetType,
    seed: int,
    autoencoder_params_path: str = None,
):
    if dataset_type == DatasetType.MNIST:
        dataset = load_mnist()
    elif dataset_type == DatasetType.FASHION_MNIST:
        dataset = load_fmnist()
    elif dataset_type == DatasetType.USPS:
        dataset = load_usps()
    elif dataset_type == DatasetType.REUTERS:
        dataset = load_reuters()

    if autoencoder_params_path is None:
        autoencoder_params_path = f"practical/DeepClustering/DeepECT/pretrained_autoencoders/{dataset['dataset_name']}_autoencoder_pretrained.pth"

    autoencoder = pretraining(
        init_autoencoder=init_autoencoder,
        autoencoder_params_path=autoencoder_params_path,
        dataset=dataset,
        seed=seed,
    )

    flat_results = flat(
        autoencoder=autoencoder,
        autoencoder_params_path=autoencoder_params_path,
        dataset_type=dataset_type,
        dataset=dataset,
        seed=seed,
    )
    hierarchical_results = hierarchical(
        autoencoder=autoencoder,
        autoencoder_params_path=autoencoder_params_path,
        dataset_type=dataset_type,
        dataset=dataset,
        seed=seed,
    )

    return flat_results, hierarchical_results


# Load the MNIST dataset and evaluate flat and hierarchical clustering
flat_results, _ = evaluate(
    init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.MNIST, seed=42
)
print(flat_results)
# evaluation(init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.USPS, seed=42)
# evaluation(init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.REUTERS, seed=42)
# evaluation(init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.FASHION_MNIST, seed=42)

# combine all results and per experiment, do pivot to aggregate the metrics over the seeds
