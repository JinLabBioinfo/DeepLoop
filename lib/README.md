It takes eight parameters:<br/>
- DeepLoopInstallPath: Path to "DeepLoop/"
- DeepLoopRefbed: Path to anchor bed, e.g. "DeepLoop/DeepLoop_models/ref/hg19_HindIII_anchor_bed/" in the test exmaples
- HiCorr_path: Path for HiCorr_output/. 
- DeepLoop_outPath: Path for DeepLoop output; where you store "chr*.denoised.anchor.to.anchor"
- chr: Genomic region chromosome
- start: Genomic region start loc
- end: Genomic region end loc, remember input a region less than 2Mb
- outPath: Path to store the heatmap png files.

If DeepLoop is installed in home directory "$myhome", outPath is current directory("./") you plan to run the script
```
bash $myhome/DeepLoop/lib/plot.sh $myhome \
                                  $myhome/DeepLoop/DeepLoop_models/ref/hg19_HindIII_anchor_bed/ \
                                  $HiCorr_path/ \
                                  $DeepLoop_outPath/ \
                                  chr11 130000000 130800000 ./ 
```
The heatmap png file named "chr11_130000000_130800000.plot.png" will be in the current directory.
