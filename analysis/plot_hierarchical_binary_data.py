import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

from analysis.hierarchical_data_utils import sample_from_hierarchical_diffusion

NUM_DESCENDANTS = 2  # 2
NUM_LEVELS = 9  # 10
E = 0.2  # 0.05, the higher, the more unique rows in data (and lower first PC)

PLOT_NUM_ROWS = None
FIGSIZE = (10, 10)
DPI = 800
TICKLABEL_FONTSIZE = 1
TITLE_FONTSIZE = 5


def cluster(mat, original_row_words=None, original_col_words=None):
    print('Clustering...')
    #
    lnk0 = linkage(pdist(mat))
    dg0 = dendrogram(lnk0,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)
    z = mat[dg0['leaves'], :]  # reorder rows
    #
    lnk1 = linkage(pdist(mat.T))
    dg1 = dendrogram(lnk1,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)

    z = z[:, dg1['leaves']]  # reorder cols
    #
    if original_row_words is None and original_col_words is None:
        return z
    else:
        row_labels = np.array(original_row_words)[dg0['leaves']]
        col_labels = np.array(original_col_words)[dg1['leaves']]
        return z, row_labels, col_labels


def to_corr_mat(data_mat):
    zscored = stats.zscore(data_mat, axis=0, ddof=1)
    res = np.dot(zscored.T, zscored)
    return res


def plot_heatmap(mat, name, ytick_labels, xtick_labels):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    title = 'Cluster Structure of\ndata sampled from hierarchical diffusion process\n' \
            'with num_vocab={} num_descendants={} num_levels={}\n' \
            '{}'.format(num_vocab, NUM_DESCENDANTS, NUM_LEVELS, name)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              aspect='equal',
              cmap=plt.get_cmap('jet'),
              interpolation='nearest')
    # xticks
    num_cols = len(mat.T)
    ax.set_xticks(np.arange(num_cols))
    ax.xaxis.set_ticklabels(xtick_labels, rotation=90, fontsize=TICKLABEL_FONTSIZE)
    # yticks
    num_rows = len(mat)
    ax.set_yticks(np.arange(num_rows))
    ax.yaxis.set_ticklabels(ytick_labels,   # no need to reverse (because no extent is set)
                            rotation=0, fontsize=TICKLABEL_FONTSIZE)
    # remove ticklines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)
    plt.show()


# vocab
num_vocab = NUM_DESCENDANTS ** NUM_LEVELS
print('num_vocab={}'.format(num_vocab))
vocab = ['w{}'.format(i) for i in np.arange(num_vocab)]
word2id = {word: n for n, word in enumerate(vocab)}


# make data_mat
data_mat = np.zeros((num_vocab, num_vocab))
for n in range(num_vocab):
    node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
    data_mat[n] = sample_from_hierarchical_diffusion(node0, NUM_DESCENDANTS, NUM_LEVELS, E)
print(data_mat)

# make random data_mat
data_mat2 = np.ones((num_vocab, num_vocab))
data_mat2 = data_mat2 - np.random.randint(0, 2, size=data_mat2.shape) * 2


for n, mat in enumerate([data_mat, data_mat2]):
    # check unique
    unique_mat = np.unique(mat, axis=0)
    # assert len(unique_mat) == len(mat)
    # corr_mat
    corr_mat = to_corr_mat(mat)
    clustered_corr_mat, row_words, col_words = cluster(corr_mat, vocab, vocab)
    # plot
    n = str(n)
    plot_heatmap(mat, 'raw data' + n, [], [])
    plot_heatmap(clustered_corr_mat, 'clustered correlations' + n, row_words, col_words)
    # pca
    pca = PCA()
    fitter = pca.fit_transform(mat)
    print(['{:.4f}'.format(i) for i in pca.explained_variance_ratio_][:10])
    break