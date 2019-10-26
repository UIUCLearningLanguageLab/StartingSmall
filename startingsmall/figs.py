import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np
from typing import List, Tuple, Union

from startingsmall import config


def human_format(num, pos):  # pos is required for formatting mpl axis ticklabels
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format(num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def make_summary_fig(summaries: list,
                     ylabel: str,
                     title: str = 'Un-titled',
                     palette_ids: List[int] = None,
                     figsize: Tuple[int, int] = None,
                     ylims: List[float] = None,
                     y_grid: bool = False,
                     plot_max_line: bool = False,
                     plot_max_lines: bool = False,
                     alternative_labels: Union[None, list] = None,
                     vlines: List[int] = None,
                     ):
    # fig
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    ax.set_xlabel('Mini Batch', fontsize=config.Figure.ax_fontsize)
    ax.set_ylabel(ylabel + '\n+/- Std Dev', fontsize=config.Figure.ax_fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.xaxis.set_major_formatter(FuncFormatter(human_format))
    if y_grid:
        ax.yaxis.grid(True)
    if ylims is not None:
        ax.set_ylim(ylims)

    # palette
    num_summaries = len(summaries)
    palette = np.asarray(sns.color_palette('hls', num_summaries))
    if palette_ids is not None:
        colors = iter(palette[palette_ids])
    else:
        colors = iter(palette)

    if alternative_labels is not None:
        alternative_labels = iter(alternative_labels)

    # plot summary
    max_ys = []
    for x, y_mean, y_std, label, n in summaries:
        max_ys.append(max(y_mean))

        if alternative_labels is not None:
            try:
                label = next(alternative_labels)
            except StopIteration:
                raise ValueError('Not enough values in ALTERNATIVE_LABELS')

        try:
            color = next(colors)
        except StopIteration:
            raise ValueError('Not enough values in PALETTE_IDS')

        ax.plot(x, y_mean, '-', linewidth=config.Figure.lw, color=color,
                label=label, zorder=3 if n == 8 else 2)
        ax.fill_between(x, y_mean + y_std, y_mean - y_std, alpha=0.5, color='grey')

    # legend
    if title:
        plt.legend(fontsize=config.Figure.legend_fontsize, frameon=False, loc='lower right', ncol=1)
    else:
        plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=1.0,
                   fontsize=config.Figure.legend_fontsize, frameon=False, loc='lower right', ncol=3)

    # max line
    if plot_max_line:
        ax.axhline(y=max(max_ys), color='grey', linestyle=':', zorder=1)
    if plot_max_lines:
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
