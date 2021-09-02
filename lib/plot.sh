#!/bin/bash
anchor_bed=$1
DeepLoop_output_dir=$2
chr=$3
start=$4
end=$5

./generate.matrix.by.DeepLoop.pl $anchor_bed/$chr.bed $DeepLoop_output_dir/$chr.denoised.anchor.to.anchor $chr $start $end {$chr}_${start}_${end} 
./plot.heatmap.r {$chr}_${start}_${end}.DeepLoop.matrix

