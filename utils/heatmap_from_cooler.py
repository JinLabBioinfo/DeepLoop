import os
import sys
import argparse
import cooler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def draw_heatmap(matrix, color_scale, ax=None, min_val=1.001, return_image=False, return_plt_im=False):
    """
    Display ratio heatmap containing only strong signals (values > 1 or 0.98th quantile)
    Args:
        matrix (:obj:`numpy.array`) : ratio matrix to be displayed
        color_scale (:obj:`int`) : max ratio value to be considered strongest by color mapping
        ax (:obj:`matplotlib.axes.Axes`) : axes which will contain the heatmap.  If None, new axes are created
        return_image (:obj:`bool`) : set to True to return the image obtained from drawing the heatmap with the generated color map
    Returns:
        ``numpy.array`` : if ``return_image`` is set to True, return the heatmap as an array
    """
    if color_scale != 0:
        breaks = np.append(np.arange(min_val, color_scale, (color_scale - min_val) / 18), np.max(matrix))
    elif np.max(matrix) < 2:
        breaks = np.arange(min_val, np.max(matrix), (np.max(matrix) - min_val) / 19)
    else:
        step = (np.quantile(matrix, q=0.98) - 1) / 18
        up = np.quantile(matrix, q=0.98) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(min_val, up, step), np.max(matrix) + 0.01)
    n_bin = 20  # Discretizes the interpolation into bins
    colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'deeploop'
    # Create the colormap
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    norm = matplotlib.colors.BoundaryNorm(breaks, 20)
    # Fewer bins will result in "coarser" colomap interpolation
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(matrix, cmap=cm, norm=norm, interpolation=None)
    if return_image:
        plt.close()
        return img.get_array()
    elif return_plt_im:
        return img



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cool_file', type=str)
    parser.add_argument('--locus', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--color_scale', type=float, default=0)
    parser.add_argument('--min_val', type=float, default=1.001)
    args = parser.parse_args()

    cool_file = args.cool_file
    locus = args.locus
    out_file = args.out_file
    color_scale = args.color_scale
    min_val = args.min_val

    if '/' in out_file:
        os.makedirs(os.path.join('/'.join(out_file.split('/')[:-1])), exist_ok=True)

    c = cooler.Cooler(cool_file)
    mat = c.matrix().fetch(locus)
    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_visible(False)
    ax = fig.add_subplot(111)
    draw_heatmap(mat, color_scale, min_val=min_val, ax=ax)
    plt.axis('off')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(out_file, bbox_inches=extent)
    plt.close()
