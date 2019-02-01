import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


from starting_small import config


def human_format(num, pos):  # pos is required for formatting mpl axis ticklabels
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def make_avg_traj_fig(x, y, traj_name):
    """
    Returns fig showing trajectory of "traj_name"
    """
    # fig
    fig, ax = plt.subplots(dpi=config.Figs.dpi)
    ax.set_xlabel('Mini Batch', fontsize=config.Figs.axlabel_fs)
    ax.set_ylabel(traj_name, fontsize=config.Figs.axlabel_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top='off', right='off')
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.yaxis.grid(True)
    # plot
    ax.plot(x, y, '-', linewidth=config.Figs.lw, color='black')
    plt.tight_layout()
    return fig
