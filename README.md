# DeepLoop
The conceptual innovation of DeepLoop is to handle systematic biases and random noises separately: we used HiCorr(https://github.com/JinLabBioinfo/HiCorr) to improve the rigor of bias correction, and then applied deep-learning techniques for noise reduction and loop signal enhancement. DeepLoop significantly improves the sensitivity, robustness, and quantitation of Hi-C loop analyses, and can be used to reanalyze most published low-depth Hi-C datasets. Remarkably, DeepLoop can identify chromatin loops with Hi-C data from a few dozen single cells

## LoopDenoise removes noise from HiCorr bias-corrected HiC data
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopDenoise.PNG" width="600" height="400">
</p>


## LoopEnhance reveals chromatin loops from a few dozens of single cell HiC data
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopEnhance_examples_sc.PNG" width="600" height="250">
</p>

# DeepLoop installation

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

# DeepLoop Usage

## Preprocessing
A detail example for Hi-C data prepossessing (downloading, mapping, filtering and HiCorr bias correction) is in https://github.com/JinLabBioinfo/DeepLoop/blob/master/preprocessing/

We recommend that users start from raw fastq data to ensure reproducibility.

## Denoise/Enhance new Hi-C data

To run either a LoopDenoise or LoopEnhance model on a HiCorr corrected dataset, please refer to the [prediction walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_prediction.ipynb)

## Visualization with [Cooler](https://github.com/open2c/cooler) and [HiGlass](http://higlass.io/)

For visualization and analysis of DeepLoop output, we recommend converting to cooler file as outlined in the [cooler walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_cooler.ipynb)

## Training new models

If you wish to train a new model, ensure you have access to a machine with a GPU and refer to the [training walkthrough notebook](https://github.com/JinLabBioinfo/DeepLoop/blob/7c742f4bf6ab57e2204c9cc21ea5f87bc60f7475/examples/walkthrough_training.ipynb)
