#!/bin/bash
DeepLoopInstallPath=$1
DeepLoopRefbed=$2
HiCorr_path=$3
DeepLoop_outPath=$4
chr=$5
start=$6
end=$7
outPath=$8

$DeepLoopInstallPath/DeepLoop/lib/generate.matrix.from_HiCorr.pl $DeepLoopRefbed/$chr.bed $HiCorr_path/anchor_2_anchor.loop.$chr $chr $start $end $outPath/${chr}_${start}_${end}
$DeepLoopInstallPath/DeepLoop/lib/generate.matrix.from_DeepLoop.pl $DeepLoopRefbed/$chr.bed $DeepLoop_outPath/$chr.denoised.anchor.to.anchor $chr $start $end $outPath/${chr}_${start}_${end}
$DeepLoopInstallPath/DeepLoop/lib/plot.multiple.r $outPath/${chr}_${start}_${end} 1 3 $outPath/${chr}_${start}_${end}.raw.matrix $outPath/${chr}_${start}_${end}.ratio.matrix $outPath/${chr}_${start}_${end}.denoise.matrix

