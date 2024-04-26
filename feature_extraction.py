import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bow(documents, feature_size):
    vectorizer = CountVectorizer(max_features=feature_size)
    features = vectorizer.fit_transform(documents).toarray()
    return features, vectorizer


def tfidf(documents, feature_size):
    vectorizer = TfidfVectorizer(max_features=feature_size)
    features = vectorizer.fit_transform(documents).toarray()
    return features, vectorizer


def w2v(documents, feature_size):
    tokenized_docs = [doc.split() for doc in documents]
    w2v_model = Word2Vec(
        tokenized_docs, vector_size=feature_size, window=5, min_count=1, workers=20
    )
    features = [
        np.array(
            [w2v_model.wv[word] for word in words if word in w2v_model.wv]
            or [np.zeros([1, feature_size])]
        )
        for words in tokenized_docs
    ]
    return features, w2v_model


def glove(documents, feature_size):
    assert feature_size in [50, 100, 200, 300]
    glove_token_size = "6B"
    glove_path = f"glove.{glove_token_size}.{feature_size}d.txt"
    glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False)
    features = [
        np.array(
            [glove_model[word] for word in doc.split() if word in glove_model]
            or [np.zeros([1, feature_size])]
        )
        for doc in documents
    ]
    return features, glove_model


def extract_features(method, documents, feature_size=1000):
    if method == "BoW":
        return bow(documents, feature_size)
    elif method == "TF-IDF":
        return tfidf(documents, feature_size)
    elif method == "Word2Vec":
        return w2v(documents, feature_size)
    elif method == "GloVe":
        return glove(documents, feature_size)
    else:
        raise ValueError("Unsupported feature extraction method.")
