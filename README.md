# DeepLoop
## LoopDenoise
![](https://raw.githubusercontent.com/JinLabBioinfo/DeepLoop/master/images/LoopDenoise_model.PNG?token=AM2XKDPSAKZMSYJE2E2URQC6S6WKC)
# Run LoopDenoise or LoopEnhance
```
./run_prediction.sh <anchor_bed_dir> <path to models> <path to anchor_to_anchor files> <output path>
#<anchor_bed_dir>: The path should contain the anchor to anchor file separated by chromosome, name as "chr1.bed", the anchor_bed file should be consistent with the file usde in HiCorr;
#<path_to_models>: The path to the LoopEnhance/LoopDenoise models;
#<path to anchor_to_anchor files>: The path to the "temp_by_chrom" from HiCorr;
#<output path>: The output path for separated LoopDenoise/LoopEnhance anchor_to_anchor files.
```
The script "run_prediction.sh" will check the cis-2M reads for input bias-corrected directory(temp_by_chrom). The LoopDenoise will be applied when the cis-2M reads is over than 250M. The LoopEnhance will be applied when the depth is lower than 250M, the LoopEnhance model trained by similar depth will be chosen.


