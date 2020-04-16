# LoopEnhance training

## LoopEnhance model
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopEnhance_model.PNG" width="900" height="160">
</p>

## LoopEnhance Training

To train a new LoopEnhance model, run the `hi_c_enhance.py` script:

```
hi_c_enhance.py <path to downsampled anchor_to_anchor files> <path to denoised anchor_to_anchor targets> <anchor_bed_dir> <experiment name>
```

You can optionally pass in additional `<path to validation replicate anchor_to_anchor files>` and `<path to validation combined anchor_to_anchor targets>`.

The script will automatically create a set of new directories storing your model experiments based on the name passed to the `<experiment name>` parameter.
