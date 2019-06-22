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


def make_summary_trajs_fig(summary_data, traj_name, title=None,
                           figsize=None, ylims=None, reverse_colors=False, y_grid=False,
                           plot_max_line=False, plot_max_lines=False, alternative_labels=None, vlines=None):
    # fig
    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        plt.title(title)
    ax.set_xlabel('Mini Batch', fontsize=config.Figs.axlabel_fs)
    ax.set_ylabel(traj_name + '\n+/- Std Dev', fontsize=config.Figs.axlabel_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    if y_grid:
        ax.yaxis.grid(True)
    if ylims is not None:
        ax.set_ylim(ylims)

    # palette
    num_summaries = len(summary_data)
    if reverse_colors:
        palette = iter(sns.color_palette('hls', num_summaries)[::-1])
    else:
        palette = iter(sns.color_palette('hls', num_summaries))

    # plot summary
    max_ys = []
    for summary_data_id, (x, mean_traj, std_traj, label, n) in enumerate(summary_data):
        max_ys.append(max(mean_traj))
        if alternative_labels is not None:
            label = next(alternative_labels)
        ax.plot(x, mean_traj, '-', linewidth=config.Figs.lw, color=next(palette),
                label=label + '\nn={}'.format(n), zorder=3 if n == 8 else 2)
        ax.fill_between(x, mean_traj + std_traj, mean_traj - std_traj, alpha=0.5, color='grey')

    # legend
    if title:
        plt.legend(fontsize=config.Figs.leg_fs, frameon=False, loc='lower right', ncol=1)
    else:
        plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=1.0,
                   fontsize=config.Figs.leg_fs, frameon=False, loc='lower right', ncol=3)

    # max line
    if plot_max_line:
        ax.axhline(y=max(max_ys), color='grey', linestyle=':', zorder=1)
    elif plot_max_lines:
        for max_y in max_ys:
            ax.axhline(y=max_y, color='grey', linestyle=':', zorder=1)
            print('y max={}'.format(max_y))
    # vertical lines
    if vlines:
        for vline in vlines:
            if vline == 0:
                continue
            print(x[-1], vline / len(vlines))
            ax.axvline(x=x[-1] * (vline / len(vlines)) , color='grey', linestyle=':', zorder=1)

    plt.tight_layout()
    return fig
