# DeepLoop
The conceptual innovation of DeepLoop is to handle systematic biases and random noises separately: we used [HiCorr](https://github.com/JinLabBioinfo/HiCorr) to improve the rigor of bias correction, and then applied deep-learning techniques for noise reduction and loop signal enhancement. DeepLoop significantly improves the sensitivity, robustness, and quantitation of Hi-C loop analyses, and can be used to reanalyze most published low-depth Hi-C datasets. 
* Zhang,S. and Plummer,D. et al._ Robust mapping of DNA loops at kilobase resolution from low depth allele-resolved or single-cell Hi-C data (under review)
## *LoopDenoise* removes noise from *HiCorr* bias-corrected Hi-C data
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopDenoise.PNG" width="900" height="400">
</p>


## *LoopEnhance* reveals chromatin loops from single cell HiC data and allele-resolved Hi-C data
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopEnhance.PNG" width="1000" height="250">
</p>

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

# *DeepLoop* Usage

## Hi-C data Preprocessing
*DeepLoop* are trained with *HiCorr* output, we have several tutorials to show how to process raw Hi-C data through *HiCorr* and *DeepLoop*. 
See [Hi-C data preprocessing](https://github.com/shanshan950/Hi-C-data-preprocess)

We highly recommend that users start from raw fastq data to ensure reproducibility.

## Run DeepLoop

To run either a LoopDenoise or LoopEnhance model on a HiCorr corrected dataset, please refer to the [prediction walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_prediction.ipynb)

## Visualization with [Cooler](https://github.com/open2c/cooler) and [HiGlass](http://higlass.io/)

For visualization and analysis of DeepLoop output, we recommend converting to cooler file as outlined in the [cooler walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_cooler.ipynb)

## Training new models

If you wish to train a new model, ensure you have access to a machine with a GPU and refer to the [training walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_training.ipynb)
