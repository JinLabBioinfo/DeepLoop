# DeepLoop
The conceptual innovation of DeepLoop is to handle systematic biases and random noises separately: we used [HiCorr](https://github.com/JinLabBioinfo/HiCorr) to improve the rigor of bias correction, and then applied deep-learning techniques for noise reduction and loop signal enhancement. DeepLoop significantly improves the sensitivity, robustness, and quantitation of Hi-C loop analyses, and can be used to reanalyze most published low-depth Hi-C datasets. 
* Zhang,S. and Plummer,D. et al._ Robust mapping of DNA loops at kilobase resolution from low depth allele-resolved or single-cell Hi-C data (under review)
## *LoopDenoise* removes noise from *HiCorr* bias-corrected Hi-C data
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopDenoise.PNG" width="900" height="380">
</p>

## *LoopEnhance* reveals chromatin loops from single cell HiC data and allele-resolved Hi-C data
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopEnhance.PNG" width="1000" height="250">
</p>


## 40 Processed Hi-C datasets by *DeepLoop* can be visualized in [website](https://hiview.case.edu/test/DeepLoop/)


# *DeepLoop* installation

DeepLoop was developed and tested using Python 3.5 and following Python packages:

* `numpy`
* `scipy`
* `pandas`
* `matplotlib`
* `opencv-python`
* `tensorflow`

The packages can be installed by running the following command:

`pip3 install -r requirements.txt`

This will also install optional visualization and analysis tools we use such as:

* `cooler`
* `jupyter`
* `higlass-python`

If you plan on training your own model you will want to use a GPU enabled version of TensorFlow to intractably long training times.  We used `tensorflow-gpu==2.3.1` but any TF2 version should work.  For prediction GPU is not necessary but it will be faster than using CPU.

# Download *DeepLoop* trained models and reference files
>We have trained a series of LoopEnhance models(depth from 100k to 250M mid-range contacts)trained with human cortex Hi-C data
>We also trained LoopDenoise model with human cortex and H9 cell line data separately. 
You can download them by:
```
cd DeepLoop/
wget --no-check-certificate https://hiview.case.edu/ssz20/tmp.HiCorr.ref/DeepLoop_models.tar.gz
tar -xvf DeepLoop_models.tar.gz
```
> After decompressing, the "DeepLoop_models/" dircetory includes "CPGZ_trained", "H9_trained" models and "ref" which includes anchor bed files for HiCorr output.
# *DeepLoop* Usage

## Hi-C data Preprocessing
- *DeepLoop* are trained with *HiCorr* output, we have several tutorials to show how to process raw Hi-C data through *HiCorr* and *DeepLoop* staring from fastq-files, bam files or "validPairs" from [HiC-Pro](https://github.com/nservant/HiC-Pro). 
See [Hi-C data preprocessing](https://github.com/shanshan950/Hi-C-data-preprocess)

- HiCorr is a fragment-based bias correction method. We highly recommend that users run HiCorr with fragment pairs instead of bin pairs, unless the experiment data achieves nucleosome resolution, e.g. MicroC. 

## Run *DeepLoop*
The format of DeepLoop input files is fragment/anchor based contact pairs from each chromosome:
<table><tr><td>anchor_id_1</td> <td>anchor_id_2</td> <td>observed_reads_count</td> <td>expected_reads_from_HiCorr</td></tr>  </table>
The output format is:
<table><tr><td>anchor_id_1</td> <td>anchor_id_2</td> <td>LoopStrength_from_DeepLoop</td></tr>  </table>
To run either a LoopDenoise or LoopEnhance model on a HiCorr corrected dataset, please refer to the [prediction walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_prediction.ipynb)

## Heatmap Visualization from DeepLoop output
```
mkdir Plots
chr=
start=
end=
DeepLoopOutPath=
$lib/generate.matrix.by.DeepLoop.pl $DeepLoopPath/DeepLoop_models/ref/${genome}_${enzyme}_anchor_bed/${chr}.bed $DeepLoopOutPath/$chr.denoised.anchor.to.anchor $chr $start $end ./Plots/${chr}_${start}_${end}
$lib/plot.heatmap.r Plots/{$chr}_${start}_${end}.DeepLoop.matrix

```
## Compatible with [HiC-Pro](https://github.com/nservant/HiC-Pro), [Cooler](https://github.com/open2c/cooler) and [HiGlass](http://higlass.io/)
- As we mentioned in section Hi-C data Preprocessing, HiCorr can take [HiC-Pro](https://github.com/nservant/HiC-Pro) output. 
- The output of HiCorr and Deeploop can be converted to [cooler](https://github.com/open2c/cooler) format, see [cooler walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_cooler.ipynb)
- You can further take converted cooler file to visulaize by  [HiGlass](http://higlass.io/)

## Training new models

If you wish to train a new model, ensure you have access to a machine with a GPU and refer to the [training walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_training.ipynb)

## About Loopcalling
DeepLoop is able to generate clean loop signals, we We will merge DeepLoop output from all the chromosomes and rank anchor pairs by "LoopStrength"(3rd column). Take confident loops from top ranked contact pairs.
