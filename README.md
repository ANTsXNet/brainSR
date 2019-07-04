# brainSR

super resolution for 3D data

* applies the model to the images in the provided directory

* places the results in a results folder with SR.nii.gz appended

```
Rscript src/applySR.R "data/*nii.gz" models/brainSR_48vgg.h5
```

script `applySR.R` should be run on CPU *not GPU*

employ `applySR_GPU.R` for GPU or on CPU if memory is an issue.

if CPU fails ( memory ) then one may need to apply the GPU version
but with different parameters than provided by default for speed.

The GPU version may require some ensembling to reduce stitching artifacts.

WIP: Results subject to change/updating at any time.

![](results/example.png?raw=true)

(left) 2mm data (center) SR applied to 2mm (right) real 1mm data



![](results/example2.png?raw=true)

(left) 2mm data (center) SR applied to 1mm to get pseudo 0.5mm (right) real 1mm data

