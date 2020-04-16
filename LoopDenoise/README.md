# LoopDenoise Training

## LoopDenoise model
<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/LoopDenoise_model.PNG" width="800" height="180">
</p>

## Data Representation

The input to the model is represented by extracting equally sized sub-matrices from the mid-range contacts.

<p align="center">
<img align="center" src="https://github.com/JinLabBioinfo/DeepLoop/blob/master/images/data_representation.png" width="400" height="400">
</p>

## Combining Replicates for Training Targets

LoopDenoise was trained by combining three sets of replicates into a training target consisting of significant and reproducible loop pixels.  To generate these training targets from your own replicates, you can run the `combine_replicates_by_p_value.py` script:

```
combine_replicates_by_p_value.py --replicates <path to replicate anchor_to_anchor files>
                                 --fulldata <path to pooled anchor_to_anchor files>
                                 --anchors <anchor_bed_dir>
                                 --output <output path>
```

There are also optional parameters such as `pmax` which sets the maximum p-value for significant loop pixels (default 0.05).

## LoopDenoise Training

To train a new LoopDenoise model, run the `hi_c_denoise.py` script:

```
hi_c_denoise.py <path to replicate anchor_to_anchor files> <path to combined anchor_to_anchor targets> <anchor_bed_dir> <experiment name>
```

You can optionally pass in additional `<path to validation replicate anchor_to_anchor files>` and `<path to validation combined anchor_to_anchor targets>`.

The script will automatically create a set of new directories storing your model experiments based on the name passed to the `<experiment name>` parameter.

Make sure you use a GPU for training, otherwise it will take far too long to fully train a model.
