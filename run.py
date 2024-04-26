import os

command = "python main.py --classifier {classifier} --reduce_method {reduce} --n_components {n_components} --extract_method {extract} --feature_size {feature} --k_fold 5 --fold {k_fold} --batch_size {batch_size} --epochs {epochs}"
gpu_prefix = "CUDA_VISIBLE_DEVICES={gpu_cnt} "

classifiers = ["SVM", "NaiveBayes", "LSTM", "TextCNN", "BERT", "RoBERTa", "OPT"]

reductions = ["PCA", "None"]  # , "LDA", "t-SNE"]
n_comps = [1, 2, 5, 10, 20, 30, 40, 50]

extracts = ["BoW", "TF-IDF", "Word2Vec"]  # , "GloVe"]
features = {
    "BoW": [50, 100, 500, 1000],
    "TF-IDF": [50, 100, 500, 1000],
    "Word2Vec": [2, 5, 10, 20, 50, 100, 300],
    "GloVe": [50, 100, 200, 300],
}
batch_size = {
    "LSTM": 64,
    "TextCNN": 64,
    "BERT": 64,
    "RoBERTa": 64,
    "OPT": 8,
}
epochs = {
    "LSTM": 40,
    "TextCNN": 40,
    "BERT": 10,
    "RoBERTa": 10,
    "OPT": 2,
}

folds = list(range(5))


def get_commands():
    commands = []
    for classifier in ["SVM", "NaiveBayes"]:
        for n_components in n_comps:
            for extract in extracts:
                for k_fold in folds:
                    cmd = command.format(
                        classifier=classifier,
                        reduce="PCA",
                        n_components=n_components,
                        extract=extract,
                        feature=features[extract][-1],
                        k_fold=k_fold,
                        batch_size=0,
                        epochs=0,
                    )
                    commands.append(cmd)

        for extract in extracts:
            for k_fold in folds:
                cmd = command.format(
                    classifier=classifier,
                    reduce="LDA",
                    n_components=1,
                    extract=extract,
                    feature=features[extract][-1],
                    k_fold=k_fold,
                    batch_size=0,
                    epochs=0,
                )
                commands.append(cmd)

        reduce = "None"
        for feature in n_comps:
            for extract in extracts:
                for k_fold in folds:
                    cmd = command.format(
                        classifier=classifier,
                        reduce=reduce,
                        n_components=0,
                        extract=extract,
                        feature=feature,
                        k_fold=k_fold,
                        batch_size=0,
                        epochs=0,
                    )
                    commands.append(cmd)
    return commands


def gpu_commands(n_gpu):
    commands = []
    gpu_cnt = 0

    for classifier in ["LSTM", "TextCNN"]:
        for extract in ["Word2Vec"]:
            for feature in features[extract][:-1]:
                for k_fold in folds:
                    cmd = command.format(
                        classifier=classifier,
                        reduce="None",
                        n_components=0,
                        extract=extract,
                        feature=feature,
                        k_fold=k_fold,
                        batch_size=batch_size[classifier],
                        epochs=epochs[classifier],
                    )
                    gpu_cmd = gpu_prefix.format(gpu_cnt=gpu_cnt)
                    gpu_cnt = (gpu_cnt + 1) % n_gpu
                    commands.append(gpu_cmd + cmd)

    for classifier in ["BERT", "RoBERTa"]:
        for k_fold in folds:
            cmd = command.format(
                classifier=classifier,
                reduce="None",
                n_components=0,
                extract="None",
                feature=0,
                k_fold=k_fold,
                batch_size=batch_size[classifier],
                epochs=epochs[classifier],
            )
            gpu_cmd = gpu_prefix.format(gpu_cnt=gpu_cnt)
            gpu_cnt = (gpu_cnt + 1) % n_gpu
            commands.append(gpu_cmd + cmd)
    return commands


if __name__ == "__main__":
    commands = get_commands()
    with open("commands.sh", "w") as f:
        for cmd in commands:
            f.write(cmd + "&\n")

    n_gpu = 8
    commands = gpu_commands(n_gpu)
    for gpu in range(n_gpu):
        with open(f"commands{gpu}.sh", "w") as f:
            for cmd in commands[gpu::n_gpu]:
                f.write(cmd + "\n")
        os.system(f"bash commands{gpu}.sh&")
