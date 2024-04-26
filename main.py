import argparse
import os

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold

import utils
from classifier import test_classifier, train_classifier
from feature_extraction import extract_features
from preprocessing import preprocess_label, preprocess_text
from reduce_dim import reduce_dimensions


def arg_parser():
    parser = argparse.ArgumentParser(description="Sentiment Analysis")
    parser.add_argument(
        "--extract_method",
        type=str,
        default="TF-IDF",
        choices=["None", "BoW", "TF-IDF", "Word2Vec", "GloVe"],
        help="Feature extraction method",
    )
    parser.add_argument("--feature_size", type=int, default=100, help="Feature size")
    parser.add_argument(
        "--reduce_method",
        type=str,
        default="PCA",
        choices=["None", "PCA", "LDA", "t-SNE"],
        help="Dimensionality reduction method",
    )
    parser.add_argument(
        "--n_components", type=int, default=50, help="Number of components"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="SVM",
        choices=["SVM", "NaiveBayes", "LSTM", "TextCNN", "BERT", "RoBERTa", "OPT"],
        help="Classifier",
    )
    parser.add_argument(
        "--dir_path", type=str, default="./temp", help="Directory path to save files"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./results",
        help="Directory path to save results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        help="Directory path to save outputs",
    )
    parser.add_argument(
        "--k_fold", type=int, default=5, help="Number of folds for cross validation"
    )
    parser.add_argument(
        "--fold", type=int, default=0, help="Fold number for cross validation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    return parser.parse_args()


def main():
    if os.path.exists("preprocessed_data.pkl"):
        preprocessed_data, preprocessed_label = utils.load_file("preprocessed_data.pkl")
    else:
        data, labels = utils.load_data("imdb_reviews.csv")
        preprocessed_data = [preprocess_text(doc) for doc in data]
        preprocessed_label = preprocess_label(labels)
        utils.dump_file(
            (preprocessed_data, preprocessed_label), "preprocessed_data.pkl"
        )

    args = arg_parser()

    extract_method = args.extract_method
    feature_size = args.feature_size
    reduce_method = args.reduce_method
    n_components = args.n_components
    classifier = args.classifier
    dir_path = args.dir_path
    result_path = args.result_path
    output_path = args.output_path
    k_fold = args.k_fold
    fold = args.fold
    batch_size = args.batch_size
    epochs = args.epochs

    if classifier in ["BERT", "RoBERTa", "OPT"]:
        per_token = True
        extract_method = "None"
    elif classifier in ["LSTM", "TextCNN"]:
        per_token = True
        assert extract_method in ["Word2Vec", "GloVe"]
        reduce_method = "None"
    else:
        per_token = False

    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    if extract_method == "None":
        features = preprocessed_data
        feature_size = 0
        reduce_method = "None"
    else:
        feature_path = f"feature_{extract_method}_{feature_size}.pkl"
        if utils.check_file_exists(feature_path, dir_path):
            features = utils.load_file(feature_path, dir_path)
        else:
            features, vectorizer = extract_features(
                extract_method, preprocessed_data, feature_size=feature_size
            )
            utils.dump_file(features, feature_path, dir_path)

        if per_token:
            reduce_method = "None"
            features = pad_sequences(features, maxlen=200, dtype="float32")

    if not per_token:
        features = np.array(
            [
                np.mean(feature, axis=0) if feature.ndim == 2 else feature
                for feature in features
            ]
        )

    if reduce_method == "None":
        n_components = 0
        reduced_features = features
    else:
        reduced_feature_name = f"reduce_{reduce_method}_{n_components}_feature_{extract_method}_{feature_size}.pkl"
        if utils.check_file_exists(reduced_feature_name, dir_path):
            reduced_features = utils.load_file(reduced_feature_name, dir_path)
        else:
            features = np.array(features)
            reduced_features = reduce_dimensions(
                reduce_method, features, preprocessed_label, n_components=n_components
            )
            utils.dump_file(reduced_features, reduced_feature_name, dir_path)

    kf = KFold(n_splits=k_fold, shuffle=True, random_state=111)

    idxs = list(kf.split(reduced_features, preprocessed_label))
    train_idx = idxs[fold][0]
    test_idx = idxs[fold][1]
    X_train = [reduced_features[i] for i in train_idx]
    y_train = [preprocessed_label[i] for i in train_idx]
    X_test = [reduced_features[i] for i in test_idx]
    y_test = [preprocessed_label[i] for i in test_idx]

    result_name = f"model_{classifier}_{extract_method}_{feature_size}_{reduce_method}_{n_components}_{k_fold}fold_{fold}.pkl"
    output_name = f"output_{classifier}_{extract_method}_{feature_size}_{reduce_method}_{n_components}_{k_fold}fold_{fold}.pkl"
    if utils.check_file_exists(result_name, result_path) and (
        fold != 0 or utils.check_file_exists(output_name, output_path)
    ):
        results = utils.load_file(result_name, result_path)
    else:
        model = train_classifier(
            classifier,
            X_train,
            y_train,
            max_features=feature_size,
            batch_size=batch_size,
            epochs=epochs,
        )

        results, predictions = test_classifier(model, X_test, y_test)

        if fold == 0:
            utils.dump_file(predictions, output_name, output_path)
        utils.dump_file(results, result_name, result_path)
    print(results)


if __name__ == "__main__":
    main()
