import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from starting_small import config


def human_format(num, pos):  # pos is required for formatting mpl axis ticklabels
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def make_summary_trajs_fig(summary_data, traj_name, figsize=None, ylims=None, reverse_colors=False):
    # fig
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('Mini Batch', fontsize=config.Figs.axlabel_fs)
    ax.set_ylabel(traj_name + '\n+/- Std Dev', fontsize=config.Figs.axlabel_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    ax.yaxis.grid(True)
    if ylims is not None:
        ax.set_ylim(ylims)
    # plot
    num_summaries = len(summary_data)
    if reverse_colors:
        palette = iter(sns.color_palette('hls', num_summaries)[::-1])
    else:
        palette = iter(sns.color_palette('hls', num_summaries))
    for x, mean_traj, std_traj, label, n in summary_data:
        ax.plot(x, mean_traj, '-', linewidth=config.Figs.lw, color=next(palette),
                label=label + '\nn={}'.format(n))
        ax.fill_between(x, mean_traj + std_traj, mean_traj - std_traj, alpha=0.5, color='grey')
    plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=1.0,
               fontsize=config.Figs.leg_fs, frameon=False, loc='lower right', ncol=2)
    plt.tight_layout()
    return fig
