### Prepare path and reference files
```
lib=lib/ # the path to lib of the Hi-C-data-preprocess 
validPair= # from HiCPro
genome= # hg19/mm10
enzyme= # restriction enzyme of the Hi-C experiment
fragbed= # enzyme fragment bed "frag_1"..., provided in HiCorr reference files, HindIII, DPNII for mm10 and hg19 are provided, other type of reference files could be generated upon request
outputname= # your outpuname or sample name
HiCorrPath= # where you put "HiCorr" file
DeepLoopPath= # go to DeepLoop/
DeepLoopBed=$DeepLoopPath/DeepLoop_models/ref/${genome}_${enzyme}_anchor_bed/
```
### Map HiCPro.all.valid.pairs to fragment pairs
```
cat $validPair | gunzip | cut -f2-7 | $lib/reads_2_cis_frag_loop.pl $fragbed 36 $outputname.loop.inward $outputname.loop.outward $outputname.loop.samestrand summary.frag_loop.read_count $outputname -
cat $validPair | gunzip | cut -f2-7 | $lib/reads_2_trans_frag_loop.pl ../test_HiCorr/test_V2/HiCorr/ref/Arima/hg19.Arima.frag.bed 36 $outputname.loop.trans - &
for file in $outputname.loop.inward $outputname.loop.outward $outputname.loop.samestrand;do
        cat $file | $lib/summary_sorted_frag_loop.pl $bed  > temp.$file &
done
cat $outputname.loop.trans | $lib/summary_sorted_trans_frag_loop.pl - > temp.$outputname.loop.trans &
wait
for file in $outputname.loop.inward $outputname.loop.outward $outputname.loop.samestrand;do
        $lib/resort_by_frag_id.pl $bed temp.$file &
done
wait
$lib/merge_sorted_frag_loop.pl temp.$outputname.loop.samestrand <(cat temp.$outputname.loop.inward | awk '{if($4>1000)print $0}') <(cat temp.$outputname.loop.outward | awk '{if($4>5000)print $0}') > frag_loop.$outputname.cis &
$lib/merge_sorted_frag_loop.pl temp.$outputname.loop.trans > frag_loop.$outputname.trans &
wait
cat frag_loop.$outputname.cis <(cat frag_loop.$outputname.cis | awk '{print $2 "\t" $1 "\t" $3 "\t" $4}') | sed s/"frag_"//g | sort -k1,2n -k2,2n | awk '{print "frag_"$1 "\t" "frag_"$2 "\t" $3 "\t" $4}' > frag_loop.$outputname.cis.tmp &
cat frag_loop.$outputname.trans <(cat frag_loop.$outputname.trans | awk '{print $2 "\t" $1 "\t" $3 "\t" $4}') | sed s/"frag_"//g | sort -k1,2n -k2,2n | awk '{print "frag_"$1 "\t" "frag_"$2 "\t" $3 "\t" $4}' > frag_loop.$outputname.trans.tmp &
wait
mv frag_loop.$outputname.cis.tmp frag_loop.$outputname.cis
mv frag_loop.$outputname.trans.tmp frag_loop.$outputname.trans
```
### Run HiCorr on cis and trans loop
```
$HiCorrPath/HiCorr ${enzyme} frag_loop.$outputname.cis frag_loop.$outputname.trans $outputname $genome
HiCorrOutputPath=`pwd`"/HiCorr_output/"
DeepLoopOutputPath=`pwd`"/DeepLoop/"
```
### Run DeepLoop on HiCorr_output
```
#Check mid-range reads:
echo `cat HiCorr_output/anchor* | awk '{sum+=$3}END{print sum/2}'` # choose model by this depth

cd $DeepLoopPath
for i in {1..22} X Y;do
  python3 predict_chromosome.py --full_matrix_dir $HiCorrOutputPath \
                                --input_name anchor_2_anchor.loop.chr${i} \
                                --h5_file DeepLoop_models/CPGZ_trained/50M.h5 \
                                --out_dir $DeepLoopOutputPath \
                                --anchor_dir $DeepLoopPath/DeepLoop_models/ref/${genome}_${enzyme}_anchor_bed/  \
                                --chromosome chr${i} --small_matrix_size 128  --step_size 128 --dummy 5
done

```
### Combine HiCorr_output and DeepLoop output
```
cd $HiCorrOutputPath
dummy=5
mkdir HiCorr_DeepLoop_comb
for i in {1..22} X Y;do
        $lib/pair.HiCorr_DeepLoop.pl HiCorr_output/anchor_2_anchor.loop.chr${i} DeepLoop/chr${i}.denoised.anchor.to.anchor $dummy > HiCorr_DeepLoop_comb/chr${i}.raw_HiCorr_DeepLoop
done &
```
##### The format for chr${i}.raw_expt_DeepLoop is 
<table><tr><td>anchor_id_1</td> <td>anchor_id_2</td> <td>observed_reads_count</td> <td>HiCorr_normalized</td> <td>DeepLoop output</td></tr>  </table>

### Plot heatmaps from HiCorr_output and DeepLoop output given chr start end and reference bed file
```
cd $HiCorrOutputPath
mkdir Plots
chr=
start=
end=
$lib/generate.raw.HiCorr.DeepLoop.matrix.pl $DeepLoopPath/DeepLoop_models/ref/${genome}_${enzyme}_anchor_bed/${chr}.bed $HiCorrOutputPath/HiCorr_DeepLoop_comb/{$chr}.raw_HiCorr_DeepLoop $chr $start $end ./Plots/$outputname.${chr}_${start}_${end}
# Plot
for file in raw.matrix HiCorr.matrix DeepLoop.matrix;do
        $HiCorrPath/bin/plot.heatmap.r Plots/$outputname.{$chr}_${start}_${end}.${file}
done
 # check the heatmaps in png files
```
