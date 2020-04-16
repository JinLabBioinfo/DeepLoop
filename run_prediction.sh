#!/bin/bash
### This script takes the "temp_by_chrom" directory as input         ####
### LoopDenoise or LoopEnhance will be performed on each chromosome  ####

#"Usage: ./run_prediction.sh <anchor_bed_file> <path to models> <path to anchor_to_anchor files> <output path>"
anchor_bed_file=$1
model_dir=$2
anchor_to_anchor_dir=$3
output_dir=$4

lib=./lib/
depth=`cat $anchor_to_anchor_dir/anchor_2_anchor.loop.* | grep -v "p_val" | awk '{sum+=$3}END{print sum/2}'`
if [[ $depth -gt 250000000 ]]
then
	echo "Input data has $depth cis-2M reads, depth is higher than 250M, recommend LoopDenoise"
	for file in `ls $anchor_to_anchor_dir/anchor_2_anchor.loop.* | grep -v "p_val"`;do
        	chr=${file#*anchor_2_anchor.loop.}
		echo Running LoopDenoise on $chr
		start=`date +%s`
        	python3 $lib/predict_chromosome.py $file $model_dir/LoopDenoise.json $model_dir/LoopDenoise.h5 $output_dir/ $anchor_bed_file $chr 128 128 5 True
		end=`date +%s`
		runtime=$((end-start))
		echo prediction on $chr took $runtime
	done
else
	echo "Input data has $depth cis-2M reads, depth is lower than 250M, recommend LoopEnhance"
	matching_depth_model=`$lib/choose_depth_matching_models.r $model_dir/ $depth`
	echo The depth matching model is $matching_depth_model
	for file in `ls $anchor_to_anchor_dir/anchor_2_anchor.loop.* | grep -v "p_val"`;do
        	chr=${file#*anchor_2_anchor.loop.}
		echo Running LoopEnhance on $chr
		start=`date +%s`
        	python3 $lib/predict_chromosome.py $anchor_to_anchor_dir/ anchor_2_anchor.loop.$chr $model_dir/$matching_depth_model.json $model_dir/$matching_depth_model.h5 $output_dir/ $anchor_bed_file $chr 128 128 5 True
		end=`date +%s`
                runtime=$((end-start))
                echo prediction on $chr took $runtime
	done	
fi
echo Prediction is done! The output is in $output_dir
