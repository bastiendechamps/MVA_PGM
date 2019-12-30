import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

from utils import load_iris_dataset, confidence_ellipse, get_cmap, custom_data
from em import EM


def model_dict(X, Z, K):
    """Run all the models and store their features into one dictionary for plotting.
    Args:
        - X : (N, d) data
        - Z : (N) labels
        - K : Number of clusters
    Returns:
        - models : Dictionary containing the means, covariance matrices (when existing) and labels of the models clusters.
    """
    # Models dictionary
    models = dict()

    # Ground truth
    models["ground truth"] = {
        "mean": np.array([X[Z == k].mean(0) for k in range(len(np.unique(Z)))]),
        "cov": None,
        "labels": Z,
    }

    # Run diagonal EM
    em_diag = EM(K)
    em_diag.fit(X)
    models["diagonal EM"] = {
        "mean": em_diag.mus,
        "cov": np.array([np.diag(em_diag.Ds[k]) for k in range(K)]),
        "labels": em_diag.labels_,
    }

    # Run general EM
    em = GaussianMixture(K)
    em.fit(X)

    # Compute reponsabilities
    gaussians = np.array(
        [
            multivariate_normal.pdf(X, em.means_[k], em.covariances_[k])
            * em.weights_[k]
            for k in range(K)
        ]
    )
    r = gaussians / gaussians.sum(0)
    models["general EM"] = {
        "mean": em.means_,
        "cov": em.covariances_,
        "labels": r.argmax(0),
    }

    # Run K-means
    km = KMeans(K)
    km.fit(X)
    models["K-means"] = {"mean": km.cluster_centers_, "cov": None, "labels": km.labels_}

    return models


def compare_models(X, Z, K=2, figsize=(7, 12), cm="viridis"):
    """Run and compare diagonal EM, general EM and K-means on Iris dataset.
    Args:
        - K : number of clusters
    """

    # Load Iris dataset
    N, d = X.shape
    true_K = len(np.unique(Z))

    # Run all the models and get the plot features
    models = model_dict(X, Z, K)

    # Plot each model's results
    features_pairs = [(i, j) for j in range(d) for i in range(j)]
    fig, axes = plt.subplots(
        ncols=len(models), nrows=len(features_pairs), figsize=figsize
    )

    for i, idx in enumerate(features_pairs):
        idx = np.array(idx)
        X_chosen = X[:, idx]

        for j, model_name in enumerate(models):
            if len(features_pairs) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            features = models[model_name]
            means = features["mean"][:, idx]
            covs = features["cov"]
            labels = features["labels"]
            K = len(np.unique(labels))

            # Color for plots
            cmap = get_cmap(K, cm)

            if covs is not None:
                covs = covs[:, idx, :][:, :, idx]

            ax.scatter(*X_chosen.T, c=labels, s=10.0, edgecolors="k", lw=0.5, cmap=cm)

            # Set subplot title
            ax.set_title(
                "{}  ({}, {})".format(model_name, idx[0], idx[1]), fontsize=10.0, y=0.96
            )
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            for k in range(K):
                mean = means[k]
                # Plot the mean
                ax.scatter(*mean, s=70.0, edgecolors="w", c=np.array([cmap(k)]))

                # Plot the ellipses when possible
                if covs is not None:
                    cov = covs[k]
                    confidence_ellipse(mean, cov, ax, color=cmap(k), facecolor=cmap(k))

    plt.show()


if __name__ == "__main__":
    # Test on Iris dataset
    X, Z = load_iris_dataset()
    K = 3
    # compare_models(X, Z, K)

    # Test on custom dataset
    X, Z = custom_data(1000)
    K = 2
    compare_models(X, Z, figsize=(10, 2))
