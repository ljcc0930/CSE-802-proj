from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE


def pca(features, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)


def lda(features, labels, n_components):
    lda = LDA(n_components=n_components)
    return lda.fit_transform(features, labels)


def tsne(features, n_components):
    tsne = TSNE(n_components=n_components, init="pca", random_state=111)
    return tsne.fit_transform(features)


def reduce_dimensions(method, features, labels=None, n_components=50):
    if method == "PCA":
        return pca(features, n_components)

    elif method == "LDA":
        assert (
            labels is not None
        ), "LDA requires labels for supervised dimensionality reduction."
        return lda(features, labels, n_components)

    elif method == "t-SNE":
        return tsne(features, n_components)

    else:
        raise ValueError("Unsupported dimensionality reduction method.")
