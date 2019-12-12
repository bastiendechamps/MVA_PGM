import csv

import numpy as np


def load(fname):
    """Load a dataset of N samples
    
    Args:
        - fname: string, the path of the dataset csv file (space separator)
    Returns:
        - X : (N, 2) np.float32 np.ndarray
        - Y : (N, 1) np.int32 np.ndarray
    """
    with open(fname, newline="", mode="r") as file:
        reader = csv.reader(file, delimiter=" ", quotechar='"')
        data = np.array(list(reader), dtype=np.float32)
    X = data[:, :2]
    Y = data[:, 2:3].astype(np.int32)
    return X, Y

