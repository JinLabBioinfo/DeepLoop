import os
import re
import cooler
import argparse
import pandas as pd
import numpy as np


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cool_file', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()

    cool_file = args.cool_file
    out_dir = args.out_dir

    bed_out_dir = os.path.join(out_dir, 'anchor_bed')  # store bin/frag/anchor references here
    loc_out_dir = os.path.join(out_dir, 'anchor_to_anchor')  # store interactions here
    os.makedirs(bed_out_dir, exist_ok=True)
    os.makedirs(loc_out_dir, exist_ok=True)

    c = cooler.Cooler(cool_file)
    print(c.info)

    chr_offset = 0  # index bins globally across genome
    chroms = c.chromnames
    bins = c.bins()[:]
    pixels = c.pixels(join=False)[:]
    pixels['count'] = pixels['count'].astype(float)
    if '_' not in str(pixels.iloc[0]['bin1_id']):
        pixels['bin1_id'] = 'A_' + pixels['bin1_id'].astype(str)
        pixels['bin2_id'] = 'A_' + pixels['bin2_id'].astype(str)
    print(pixels)
    for chrom in sorted_nicely(chroms):
        print(chrom)
        chr_bins = bins[bins['chrom'] == chrom]
        chr_bins.loc[:, 'bin_id'] = np.arange(1, len(chr_bins) + 1) + chr_offset
        chr_bins.loc[:, 'bin_id'] = 'A_' + chr_bins['bin_id'].astype(str)
        chr_bins[['chrom', 'start', 'end', 'bin_id']].to_csv(os.path.join(bed_out_dir, '%s.bed' % chrom), sep='\t', header=False, index=False)
        chr_pixels = pixels.loc[(pixels['bin1_id'].isin(chr_bins['bin_id'])) & (pixels['bin2_id'].isin(chr_bins['bin_id']))]
        chr_pixels.to_csv(os.path.join(loc_out_dir, 'anchor_2_anchor.loop.%s' % chrom), sep='\t', header=False, index=False)
        chr_offset += len(chr_bins)  # next bin indices will start from previous chrom
