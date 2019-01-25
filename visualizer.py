from sklearn.decomposition import TruncatedSVD
from yellowbrick.text import TSNEVisualizer

from generate import LoadData, Representation
import matplotlib.pyplot as plt

X_train, y_train, _, _ = LoadData.load_data("dialect")
X_train = Representation.get_representation("bow").fit_transform(X_train)


def truncatedSVD_graph(X_train, y_train):
    svd = TruncatedSVD(n_components=2)

    X_coords = svd.fit_transform(X_train)

    data1 = list(list(map(lambda x: x[0], list(filter(lambda x: x[1] == 1, list(zip(X_coords, y_train)))))))
    data2 = list(list(map(lambda x: x[0], list(filter(lambda x: x[1] == 2, list(zip(X_coords, y_train)))))))

    data = (data1, data2)
    labels = ("moldavian", "romanian")
    colors = ("red", "blue")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

    for coords, label, color in zip(data, labels, colors):
        print(coords)
        x = list(map(lambda x: x[0], coords))
        y = list(map(lambda x: x[1], coords))
        ax.scatter(x, y, c=color, edgecolor="none", label=label)

    plt.title("Bow representation")
    plt.legend(loc=2)
    plt.show()

def TSNE_graph(X_train, y_train):
    tsne = TSNEVisualizer()
    tsne.fit(X_train, y_train)
    tsne.poof()

TSNE_graph(X_train, y_train)