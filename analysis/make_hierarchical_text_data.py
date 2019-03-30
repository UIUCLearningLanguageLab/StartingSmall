import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from scipy import stats

E = 0.01
NUM_SAMPLES = 1000
NUM_DESCENDANTS = 3  # 2
NUM_LEVELS = 6  # 9


PLOT_NUM_ROWS = None
FIGSIZE = (10, 10)
DPI = 200
TICKLABEL_FONTSIZE = 20
AXLABEL_FONTSIZE = 20
TITLE_FONTSIZE = 10


def cluster(mat):
    print('Clustering...')
    lnk0 = linkage(pdist(mat))
    dg0 = dendrogram(lnk0,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)

    z = mat[dg0['leaves'], :]  # reorder rows
    # top dendrogram
    lnk1 = linkage(pdist(mat.T))
    dg1 = dendrogram(lnk1,
                     ax=None,
                     color_threshold=-1,
                     no_labels=True,
                     no_plot=True)

    z = z[:, dg1['leaves']]  # reorder cols to match leaves of dendrogram
    return z


def plot_heatmap(mat, name):
    fig, ax_heatmap = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    title = 'Cluster Structure of\ndata sampled from hierarchical diffusion process\n' \
            'with num_samples={} num_descendants={} num_levels={}\n' \
            '{}'.format(NUM_SAMPLES, NUM_DESCENDANTS, NUM_LEVELS, name)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    # heatmap
    print('Plotting heatmap...')
    ax_heatmap.imshow(mat,
                      aspect='equal',
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest')
    plt.show()


def sample_using_hierarchical_diffusion(num_levels=NUM_LEVELS, num_descendants=NUM_DESCENDANTS, e=E):
    node0 = -1 if np.random.binomial(n=1, p=0.5) else 1
    nodes = [node0]
    for level in range(num_levels):
        candidate_nodes = nodes * num_descendants
        nodes = [node if np.random.binomial(n=1, p=1-e) else -node for node in candidate_nodes]
    return nodes


# make data_mat
data_mat = np.zeros((NUM_SAMPLES, NUM_DESCENDANTS**NUM_LEVELS))
for n in range(NUM_SAMPLES):
    data_mat[n] = sample_using_hierarchical_diffusion()

# make random data_mat
data_mat2 = np.ones((NUM_SAMPLES, NUM_DESCENDANTS**NUM_LEVELS))
data_mat2 = data_mat2 - np.random.randint(0, 2, size=data_mat2.shape) * 2


for n, mat in enumerate([data_mat, data_mat2]):
    zscored = stats.zscore(mat, axis=0, ddof=1)
    corr_z = np.dot(zscored.T, zscored)

    # plot
    n = str(n)
    plot_heatmap(mat, 'raw data' + n)
    # plot_heatmap(cluster(mat), 'clustered raw data' + n)
    # plot_heatmap(zscored, 'zscored data' + n)
    # plot_heatmap(cluster(zscored), 'clustered zscored data' + n)
    # plot_heatmap(corr_z, 'correlations z' + n)
    plot_heatmap(cluster(corr_z), 'clustered correlations z' + n)
