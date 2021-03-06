library( ANTsRNet )
library( tensorflow )
library( keras )
Sys.setenv(  CUDA_VISIBLE_DEVICES=-1 )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript applySR.R inputFiles modelFile\n" )
  stop( helpMessage )
  } else {
  inputFileName <- Sys.glob( args[1] )
  modelFile <- args[2]
  }

# mdl = load_model_hdf5( modelFile )
for ( x in 1:length( inputFileName ) ) {
  outfn = tools::file_path_sans_ext( basename( inputFileName[ x ] ), T )
  print( paste( "Apply SR to:", inputFileName[ x ]  ) )
  img = antsImageRead( inputFileName[ x ] )
  mdl = createDeepBackProjectionNetworkModel3D( c( dim( img ),  1 ),
     numberOfOutputs = 1, numberOfBaseFilters = 64,
     numberOfFeatureFilters = 256, numberOfBackProjectionStages = 7,
     convolutionKernelSize = rep( 3, 3 ),
     strides = rep( 2, 3 ),
     numberOfLossFunctions = 1 )
  load_model_weights_hdf5( mdl, modelFile )
  augger <- function( image, trans, interpolation = 'linear' ) {
    moveX1 = createAntsrTransform(dimension=3, type="Euler3DTransform", translation=trans )
    shift1 = applyAntsrTransform(transform=moveX1, data=image, reference=image, interpolation=interpolation )
    srout = applySuperResolutionModel( shift1, mdl, targetRange = c( 127.5, -127.5) )
    return( applyAntsrTransform( transform = invertAntsrTransform( moveX1 ), data=srout, reference=srout, interpolation=interpolation  ) )
    }
  print('Begin')
  iaug = list()
  for ( k in 1:10 )
    iaug[[k]] = augger( img, antsGetSpacing( img ) * rnorm(3) )
  wt = 0.2
  mysr = antsAverageImages( iaug, normalize=FALSE )
  mysr2 = mysr * (1-wt) + iMath( mysr, "Sharpen" ) * wt
  dir.create( file.path( './', 'results'), showWarnings = FALSE)
  antsImageWrite( mysr, paste0( './results/', outfn, "_SR.nii.gz" ) )
  antsImageWrite( mysr2, paste0( './results/', outfn, "_SR2.nii.gz" ) )
  # antsImageWrite( mysrf, paste0( './results/', outfn, "_SRf.nii.gz" ) )
}
