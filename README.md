# DeepLoop
The conceptual innovation of DeepLoop is to handle systematic biases and random noises separately: we used HiCorr(https://github.com/JinLabBioinfo/HiCorr) to improve the rigor of bias correction, and then applied deep-learning techniques for noise reduction and loop signal enhancement. DeepLoop significantly improves the sensitivity, robustness, and quantitation of Hi-C loop analyses, and can be used to reanalyze most published low-depth Hi-C datasets. Remarkably, DeepLoop can identify chromatin loops with Hi-C data from a few dozen single cells

## LoopDenoise removes noise from HiCorr bias-corrected HiC data
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopDenoise.example.PNG" width="600" height="400">
</p>


## LoopEnhance reveals chromatin loops from a few dozens of single cell HiC data
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopEnhance_examples_sc.PNG" width="600" height="250">
</p>

# DeepLoop installation



# DeepLoop Usage
A detail example for HiC data prepossessing (downloading, mapping, filtering and HiCorr bias correction) is in https://github.com/JinLabBioinfo/DeepLoop/blob/master/preprocessing/

The "temp_by_chrom" directory in the output of HiCorr bias correction contains loop anchor pairs with raw reads and expected reads. 
The format is <loop_anchor1> <loop_anchor2> <raw_reads> <expected reads>  
The script "run_prediction.sh" will check the cis-2M reads for input bias-corrected directory(temp_by_chrom). The LoopDenoise will be applied when the cis-2M reads is over than 250M. The LoopEnhance will be applied when the depth is lower than 250M, the LoopEnhance model trained by similar depth will be chosen.
```
./run_prediction.sh <anchor_bed_dir> <path to models> <path to anchor_to_anchor files> <output path>
<anchor_bed_dir>: The path should contain the anchor to anchor file separated by chromosome, name as "chr1.bed", the anchor_bed file should be consistent with the file usde in HiCorr;
<path_to_models>: The path to the LoopEnhance/LoopDenoise models;
<path to anchor_to_anchor files>: The path to the "temp_by_chrom" from HiCorr;
<output path>: The output path for separated LoopDenoise/LoopEnhance anchor_to_anchor files.
```



