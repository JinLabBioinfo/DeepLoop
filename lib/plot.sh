#!/bin/bash
chr=$1
start=$2
end=$3

./generate.raw.HiCorr.DeepLoop.matrix.pl anchor_bed/$chr.bed HK2662/$chr.raw_HiCorr_DeepLoop $chr $start $end ./Plots/HK2662.{$chr}_${start}_${end} &
./generate.raw.HiCorr.DeepLoop.matrix.pl anchor_bed/$chr.bed HK2663/$chr.raw_HiCorr_DeepLoop $chr $start $end ./Plots/HK2663.{$chr}_${start}_${end} &
./generate.raw.HiCorr.DeepLoop.matrix.pl anchor_bed/$chr.bed merge/$chr.raw_HiCorr_DeepLoop $chr $start $end ./Plots/merge.{$chr}_${start}_${end} &
wait
for file in raw.matrix HiCorr.matrix DeepLoop.matrix;do
        ./plot.heatmap.r Plots/HK2662.{$chr}_${start}_${end}.${file}
	./plot.heatmap.r Plots/HK2663.{$chr}_${start}_${end}.${file}
	./plot.heatmap.r Plots/merge.{$chr}_${start}_${end}.${file}
done
