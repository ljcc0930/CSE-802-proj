import os
import pickle as pkl
import pandas as pd


def load_data(filepath):
    data = pd.read_csv(filepath)

    reviews = data["review"].values
    sentiments = data["sentiment"].values

    return reviews, sentiments


def check_file_exists(file_name, path="."):
    return os.path.exists(os.path.join(path, file_name))


def load_file(file_name, path="."):
    with open(os.path.join(path, file_name), "rb") as f:
        return pkl.load(f)


def dump_file(data, file_name, path="."):
    with open(os.path.join(path, file_name), "wb") as f:
        pkl.dump(data, f)


def save_fig(plt, fig_name, path=".", dpi=300, file_type="pdf"):
    plt.savefig(
        os.path.join(path, f"{fig_name}.{file_type}"), dpi=dpi, bbox_inches="tight"
    )


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
