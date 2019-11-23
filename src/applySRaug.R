library( ANTsRNet )
library( tensorflow )
library( keras )
args <- commandArgs( trailingOnly = TRUE )
patchBased = FALSE
nAugmentations = 5
blendingWeight = 0.8
if ( length( args ) < 2 )
  {
  helpMessage <- paste0( "Usage:
  Rscript applySR.R inputFiles modelFile
    optional-patch-based-bool optional-nAugmentations optional-blendingWeight
    where 0<= blendingWeight <= 1 \n
    User should set patch-based to TRUE if using GPU and FALSE for CPU. \n
    User should set nAugmentations >=5 to limit bias or artifacts. " )
  stop( helpMessage )
  } else {
  inputFileName <- Sys.glob( args[1] )
  modelFile <- args[2]
  if ( length( args ) >= 3 ) patchBased = as.logical( args[3] )
  if ( length( args ) >= 4 ) nAugmentations = as.numeric( args[4] )
  if ( length( args ) >= 5 ) blendingWeight = as.numeric( args[5] )
  }
cat("Options =
  <input> modelFile patchBased nAugmentations blendingWeight\n")

# mdl = load_model_hdf5( modelFile )
for ( x in 1:length( inputFileName ) ) {
  outfn = tools::file_path_sans_ext( basename( inputFileName[ x ] ), T )
  print( paste( "Apply SR to:", inputFileName[ x ]  ) )
  img = antsImageRead( inputFileName[ x ] )
  inputsize = list( NULL, NULL, NULL, 1 )
  if ( patchBased ) inputsize = list( 24, 24, 24,  1 )
  mdl = createDeepBackProjectionNetworkModel3D( inputsize,
     numberOfOutputs = 1, numberOfBaseFilters = 64,
     numberOfFeatureFilters = 256, numberOfBackProjectionStages = 7,
     convolutionKernelSize = rep( 3, 3 ),
     strides = rep( 2, 3 ),
     numberOfLossFunctions = 1 )
  load_model_weights_hdf5( mdl, modelFile )
  augger <- function( image, trans, interpolation = 'linear', patchBased = FALSE ) {
    moveX1 = createAntsrTransform(dimension=3, type="Euler3DTransform", translation=trans )
    shift1 = applyAntsrTransform(transform=moveX1, data=image, reference=image, interpolation=interpolation )
    if ( ! patchBased ) srout = applySuperResolutionModel( shift1, mdl, targetRange = c( -127.5, 127.5) )
    if (   patchBased ) srout = applySuperResolutionModelPatch( shift1, mdl,
      targetRange = c( -127.5, 127.5 ), lowResolutionPatchSize=24, strideLength = 7 )
    return( applyAntsrTransform( transform = invertAntsrTransform( moveX1 ),
      data = srout, reference = srout, interpolation = interpolation  ) )
    }
  print('Begin')
  iaug = list()
  for ( k in 1:nAugmentations ) {
    cat( paste0("...",k,"") )
    scl = 0.5
    iaug[[k]] = augger( img, antsGetSpacing( img ) * rnorm(3) * scl )
    }
  wt = blendingWeight
  mysr = antsAverageImages( iaug, normalize=FALSE )
  mysr2 = mysr * (1-wt) + iMath( mysr, "Sharpen" ) * wt
  dir.create( file.path( './', 'results'), showWarnings = FALSE)
  antsImageWrite( mysr, paste0( './results/', outfn, "_SR.nii.gz" ) )
  }
