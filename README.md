# brainSR

super resolution for 3D data

* applies the model to the images in the provided directory

* places the results in a results folder with SR.nii.gz appended

```
Rscript src/applySR.R "data/*nii.gz" models/brainSR_48vgg.h5
```

script `applySR.R` should be run on CPU *not GPU*

employ `applySR_GPU.R` for GPU or on CPU if memory is an issue.
