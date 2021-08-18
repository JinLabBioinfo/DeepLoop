import os
import re
import sys
import math
import argparse
import cooler
import swifter
import matplotlib
matplotlib.use('Agg')  # necessary when plotting without $DISPLAY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import combinations
from multiprocessing import Pool
from scipy.sparse import coo_matrix, csr_matrix


def anchor_list_to_dict(anchors):
    """
    Converts the array of anchor names to a dictionary mapping each anchor to its chromosomal index

    Args:
        anchors (:obj:`numpy.array`) : array of anchor name values

    Returns:
        `dict` : dictionary mapping each anchor to its index from the array
    """
    anchor_dict = {}
    for i, anchor in enumerate(anchors):
        anchor_dict[anchor] = i
    return anchor_dict


def anchor_to_locus(anchor_dict):
    """
    Function to convert an anchor name to its genomic locus which can be easily vectorized

    Args:
        anchor_dict (:obj:`dict`) : dictionary mapping each anchor to its chromosomal index

    Returns:
        `function` : function which returns the locus of an anchor name
    """
    def f(anchor):
        return anchor_dict[anchor]
    return f


def sorted_nicely(l):
    """
    Sorts an iterable object according to file system defaults
    Args:
        l (:obj:`iterable`) : iterable object containing items which can be interpreted as text
    Returns:
        `iterable` : sorted iterable
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def open_anchor_to_anchor(filename, col_names, cooler_col='count'):
    '''
    Read a tab delimited anchor to anchor file as a DataFrame
    Args:
        filename (:obj:`str`) : full path to anchor to anchor file

    Returns:
        ``pandas.DataFrame``: if reading a normalized anchor to anchor file, columns are ``a1 a2 obs exp ratio``
        and if reading a denoised or enhanced anchor to anchor file, columns are ``a1 a2 ratio``
    '''
    df = pd.read_csv(filename, sep='\t')
    n_cols = len(df.columns)

    df = pd.read_csv(filename,
                     sep='\t',
                     names=col_names)
    if 'obs' in col_names and 'exp' in col_names:
        df['ratio'] = (df['obs'] + 5) / (df['exp'] + 5)

    df = df[['a1', 'a2', cooler_col]]
    return df


def draw_heatmap(matrix, color_scale, ax=None, return_image=False):
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
        breaks = np.append(np.arange(1.001, color_scale, (color_scale - 1.001) / 18), np.max(matrix))
    elif np.max(matrix) < 2:
        breaks = np.arange(1.001, np.max(matrix), (np.max(matrix) - 1.001) / 19)
    else:
        step = (np.quantile(matrix, q=0.98) - 1) / 18
        up = np.quantile(matrix, q=0.98) + 0.011
        if up < 2:
            up = 2
            step = 0.999 / 18
        breaks = np.append(np.arange(1.001, up, step), np.max(matrix))

    n_bin = 20  # Discretizes the interpolation into bins
    colors = ["#FFFFFF", "#FFE4E4", "#FFD7D7", "#FFC9C9", "#FFBCBC", "#FFAEAE", "#FFA1A1", "#FF9494", "#FF8686",
              "#FF7979", "#FF6B6B", "#FF5E5E", "#FF5151", "#FF4343", "#FF3636", "#FF2828", "#FF1B1B", "#FF0D0D",
              "#FF0000"]
    cmap_name = 'my_list'
    # Create the colormap
    cm = matplotlib.colors.LinearSegmentedColormap.from_list(
        cmap_name, colors, N=n_bin)
    norm = matplotlib.colors.BoundaryNorm(breaks, 20)
    # Fewer bins will result in "coarser" colomap interpolation
    if ax is None:
        _, ax = plt.subplots()
    img = ax.imshow(matrix, cmap=cm, norm=norm, interpolation='nearest')
    if return_image:
        plt.close()
        return img.get_array()


def get_chromosome_from_filename(filename):
    """
    Extract the chromosome string from any of the file name formats we use
    Args:
        filename (:obj:`str`) : name of anchor to anchor file
    Returns:
        Chromosome string of form chr<>
    """
    chr_index = filename.find('chr')  # index of chromosome name
    if chr_index == 0:  # if chromosome name is file prefix
        return filename[:filename.find('.')]
    file_ending_index = filename.rfind('.')  # index of file ending
    if chr_index > file_ending_index:  # if chromosome name is file ending
        return filename[chr_index:]
    else:
        return filename[chr_index: file_ending_index]


def get_anchor_length(bins, key='a1'):
    # much faster to access dict values from anchor keys than from a DataFrame (https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe)
    bins_dict = bins.to_dict(orient='index')
    def f(row):
        # access rows of dict instead of using .loc --> ~10x speed up!
        old_bin1 = bins_dict[int(row[key + '_id'])]
        return int(old_bin1['end'] - old_bin1['start'])
    return f


def anchor_to_bin(bins, key='a1', offset=0):
    bins_dict = bins.to_dict(orient='index')
    def f(row):
        old_bin1 = bins_dict[int(row[key + '_id'])]
        return int(old_bin1['start'] / bin_size) + offset
    return f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor_dir', type=str)
    parser.add_argument('--loop_dir', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--bin_size', type=int, default=-1)
    parser.add_argument('--min_ratio', type=float, default=0)
    parser.add_argument('--force_bin_size', type=bool, default=False)
    parser.add_argument('--zoomify', type=bool, default=False)
    parser.add_argument('--multires_outfile', type=str, default=None)
    parser.add_argument('--col_names','--list', nargs='+', help='names of value columns', default=['a1', 'a2', 'obs', 'exp'])
    parser.add_argument('--cooler_col', type=str, help='names of value column that will be saved to cooler file', default=['ratio'])
    parser.add_argument('--save_obs', action='store_true')
    parser.add_argument('--single_chrom', type=str)

    args = parser.parse_args()

    anchor_dir = args.anchor_dir
    loop_dir = args.loop_dir
    out_file = args.out_file
    bin_size = args.bin_size
    min_ratio = args.min_ratio
    force_bin_size = args.force_bin_size
    zoomify = args.zoomify
    multires_outfile = args.multires_outfile
    col_names = args.col_names
    cooler_col = args.cooler_col
    save_obs = args.save_obs
    single_chrom_name = args.single_chrom
    if single_chrom_name is not None:
        single_chrom = True
    if len(out_file.split('/')) > 1:
        os.makedirs('/'.join(out_file.split('/')[:-1]), exist_ok=True)

    bin_files = os.listdir(anchor_dir)

    genome_bins = pd.DataFrame()
    genome_anchors = pd.DataFrame()
    genome_pixels = pd.DataFrame()
    print(out_file)

    if os.path.isdir(loop_dir):
        chrom_files = os.listdir(loop_dir)
    else:
        chrom_files = [loop_dir]


    for file in tqdm(sorted_nicely(chrom_files)):
        if 'chr' not in file:
            continue
        chr_name = get_chromosome_from_filename(file)
        if single_chrom:
            if single_chrom_name != chr_name:
                continue
        bin_file = os.path.join(anchor_dir, chr_name + '.bed')

        bins = pd.read_csv(bin_file, sep='\t', names=['chrom', 'start', 'end', 'anchor'])
        anchor_to_int = lambda s: int(s.split('_')[-1]) - 1  # convert anchor names to integer IDs
        bins['anchor'] = bins['anchor'].apply(anchor_to_int)
        bins['weight'] = 1
        chr_offset = bins['anchor'].min()  # number of anchors preceding this chromosome
        chr_bin_offset = int(bins['start'].min() / bin_size)  # number of bins
        if single_chrom:
            bins['anchor'] = bins['anchor'] - chr_offset
        if len(genome_anchors) > 0:
            chr_offset = int(genome_anchors['anchor'].max())
            chr_bin_offset = int(genome_anchors['start'].max() / bin_size)
        if bin_size != -1:
            uniform_bins = pd.DataFrame()
            chr_len = bins['end'].max()
            if not force_bin_size:
                n_bins = len(bins)
                starts = np.arange(0, n_bins * bin_size, bin_size)
                ends = starts + bin_size
            else:
                n_bins = int(chr_len / bin_size)
                starts = np.arange(0, n_bins * bin_size, bin_size)
                ends = starts + bin_size
            uniform_bins['start'] = starts
            uniform_bins['end'] = ends
            uniform_bins['weight'] = 1
            genome_bins = pd.concat([genome_bins, uniform_bins])
            genome_anchors = pd.concat([genome_anchors, bins])
        else:
            genome_bins = pd.concat([genome_bins, bins])
            genome_anchors = pd.concat([genome_anchors, bins])

        pixels = open_anchor_to_anchor(os.path.join(loop_dir, file), col_names, cooler_col=cooler_col)
        #pixels = pd.read_csv(os.path.join(loop_dir, file), sep='\t', names=['a1', 'a2'] + col_names)
        pixels = pixels[pixels[cooler_col] >= min_ratio]
        try:
            pixels['a1'] = pixels['a1'].str.extract('(\d+)').astype(int)
            pixels['a2'] = pixels['a2'].str.extract('(\d+)').astype(int)
        except Exception as e:
            print(pixels)
        pixels = pixels[pixels['a1'] <= pixels['a2']]
        if single_chrom:
            pixels['a1'] = pixels['a1'] - chr_offset
            pixels['a2'] = pixels['a2'] - chr_offset
        if force_bin_size:
            # convert each anchor id into a new bin id; init all to 0
            anchor_map = genome_anchors.set_index('anchor')
            pixels['new_bin1'] = 0
            pixels['new_bin2'] = 0
            pixels['bin1_length'] = 0
            pixels['bin2_length'] = 0
            print('Mapping a1 to new bins...')
            pixels['new_bin1'] = pixels.swifter.apply(anchor_to_bin(anchor_map, key='a1', offset=chr_bin_offset), axis=1)
            print('Computing a1 original lengths to find overlap with new bins...')
            pixels['bin1_length'] = pixels.swifter.apply(get_anchor_length(anchor_map, key='a1'), axis=1)
            print('Mapping a2 to new bins...')
            pixels['new_bin2'] = pixels.swifter.apply(anchor_to_bin(anchor_map, key='a2', offset=chr_bin_offset), axis=1)
            print('Computing a2 original lengths to find overlap with new bins...')
            pixels['bin2_length'] = pixels.swifter.apply(get_anchor_length(anchor_map, key='a2'), axis=1)
            pixels['bin1_overlap'] = pixels['bin1_length'] / bin_size
            pixels['bin2_overlap'] = pixels['bin2_length'] / bin_size
            pixels['bin1_overlap'] = pixels['bin1_overlap'].astype(int)
            pixels['bin2_overlap'] = pixels['bin2_overlap'].astype(int)
            overlap_counts = pd.concat([pixels['bin1_overlap'], pixels['bin2_overlap']]).unique()
            print(sorted(overlap_counts))

            def smear_pixels(pixels, bin1_overlap, bin2_overlap):
                tmp_new_pixels = [pd.DataFrame()]
                bin1_mask = pixels['bin1_overlap'] == bin1_overlap
                bin2_mask = pixels['bin2_overlap'] == bin2_overlap
                a1_overlap = min(bin1_overlap, 10)  # clip overlap to reasonable range
                a2_overlap = min(bin2_overlap, 10)
                if a1_overlap * a2_overlap < 64:
                    for a1_offset in range(-int(a1_overlap / 2), int(a1_overlap / 2)):
                        if abs(a1_offset) > 0:
                            for a2_offset in range(-int(a2_overlap / 2), int(a2_overlap / 2)):
                                if abs(a2_offset) > 0:
                                    new_a1_pixels = pixels.loc[bin1_mask].copy(deep=True)
                                    new_a2_pixels = pixels.loc[bin2_mask].copy(deep=True)
                                    new_a1_pixels.loc[:, 'new_bin1'] = new_a1_pixels['new_bin1'] + a1_offset
                                    new_a2_pixels.loc[:, 'new_bin2'] = new_a2_pixels['new_bin2'] + a2_offset
                                    new_a1_pixels = new_a1_pixels[new_a1_pixels['new_bin1'] <= new_a1_pixels['new_bin2']]
                                    new_a2_pixels = new_a2_pixels[new_a2_pixels['new_bin1'] <= new_a2_pixels['new_bin2']]
                                    tmp_new_pixels += [new_a1_pixels, new_a2_pixels]
                    tmp_new_pixels = pd.concat(tmp_new_pixels)
                #print('Finished smearing overlap (%d, %d) pixels...' % (bin1_overlap, bin2_overlap))
                return tmp_new_pixels

            results = []
            print('Smearing pixels with multiprocessing...')
            with Pool(20) as pool:
                for bin1_overlap in sorted(overlap_counts):
                    for bin2_overlap in sorted(overlap_counts):
                        results += [pool.apply_async(smear_pixels, args=(pixels, bin1_overlap, bin2_overlap))]
                tmp_new_pixels = []
                for res in results:
                    try:
                        tmp_new_pixels += [res.get(timeout=300)]
                    except Exception as e:
                        print(e)
                        pass
            tmp_new_pixels = pd.concat(tmp_new_pixels)
            print(tmp_new_pixels)
            pixels = pd.concat([pixels, tmp_new_pixels])
            del tmp_new_pixels
            min_bin = chr_bin_offset
            max_bin = chr_bin_offset + len(uniform_bins)
            tmp_new_pixels = tmp_new_pixels[(tmp_new_pixels['new_bin1'] > min_bin) & (tmp_new_pixels['new_bin1'] < max_bin)]
            tmp_new_pixels = tmp_new_pixels[(tmp_new_pixels['new_bin2'] > min_bin) & (tmp_new_pixels['new_bin2'] < max_bin)]
            pixels = pd.concat([pixels, tmp_new_pixels])
            # assuming we are binning to lower resolution, aggregate duplicate loci
            pixels = pixels[['new_bin1', 'new_bin2', cooler_col]]
            pixels = pixels.groupby(['new_bin1', 'new_bin2'])[cooler_col].max().reset_index()
            pixels.rename(columns={'new_bin1': 'a1', 'new_bin2': 'a2'}, inplace=True)
            pixels.drop_duplicates(subset=['a1', 'a2'], inplace=True, ignore_index=True)
            pixels.sort_values(by=['a1', 'a2'], inplace=True)
            print(pixels)
            if len(pixels) > 0:
                max_anchor = max(pixels['a1'].max(), pixels['a2'].max()) + 1
                anchor_dict = anchor_list_to_dict(np.arange(chr_bin_offset, chr_bin_offset + max_anchor))  # convert to anchor --> index dictionary
                rows = np.vectorize(anchor_to_locus(anchor_dict))(
                    pixels['a1'].values)  # convert anchor names to row indices
                cols = np.vectorize(anchor_to_locus(anchor_dict))(
                    pixels['a2'].values)  # convert anchor names to column indices

                matrix = csr_matrix((pixels['count'], (rows, cols)),
                                    shape=(max_anchor, max_anchor))
                matrix = (matrix + matrix.transpose())
            else:
                matrix = csr_matrix((len(uniform_bins), len(uniform_bins)))
            preview_slice = slice(512, 512 + 256)
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            draw_heatmap(matrix[preview_slice, preview_slice].A, 0, ax=axs[0])
            #axs[0].imshow(matrix[preview_slice, preview_slice].A, cmap='Reds')
            axs[1].imshow(np.log1p(matrix[preview_slice, preview_slice].A), cmap='jet')
            plt.savefig('heatmaps/forced_bins_test_%s.png' % chr_name)
            plt.close()

        genome_pixels = pd.concat([genome_pixels, pixels])
    genome_pixels.drop_duplicates(subset=['a1', 'a2'], inplace=True, ignore_index=True)
    genome_pixels['a1'] = genome_pixels['a1'].astype(int)
    genome_pixels['a2'] = genome_pixels['a2'].astype(int)
    genome_pixels = genome_pixels[genome_pixels['a1'] <= genome_pixels['a2']]
    genome_pixels.sort_values(by=['a1', 'a2'], inplace=True)
    print(genome_pixels)
    print(genome_pixels.dtypes)
    genome_pixels.rename(columns={'a1': 'bin1_id', 'a2': 'bin2_id', cooler_col: 'count'}, inplace=True)
    print('Saving cooler...')
    c = cooler.create_cooler(cool_uri=out_file, bins=genome_bins, pixels=genome_pixels[['bin1_id', 'bin2_id', 'count']], assembly='hg19', dtypes={'count': float})

    if zoomify and multires_outfile is not None:
        print('Zoomifying cooler...')
        cooler.zoomify_cooler(out_file,
                      outfile=multires_outfile,
                      chunksize=2048, nproc=20, balance=False, agg={cooler_col: np.sum if save_obs else np.max},
                      resolutions=[10000, 20000, 40000])
