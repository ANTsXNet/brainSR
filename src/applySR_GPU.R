library( ANTsRNet )
library( tensorflow )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if ( length( args ) < 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript applySR_GPU.R inputFiles modelFile optional-stride-length optional-do-crop optional-do-perm\n" )
  stop( helpMessage )
  } else {
  inputFileName <- Sys.glob( args[1] )
  modelFile <- args[2]
  }
strideLength = 14
doCrop = TRUE
doPerm = FALSE
if ( length( args ) > 2 ) strideLength = as.numeric( as.character( args[3] ) )
if ( length( args ) > 3 ) doCrop = as.logical( as.character( args[4] ) )
if ( length( args ) > 4 ) doPerm = as.logical( as.character( args[5] ) )
print( paste( "Options:", doCrop, doPerm ) )
# mdl = load_model_hdf5( modelFile )
for ( x in 1:length( inputFileName ) ) {
  outfn = tools::file_path_sans_ext( basename( inputFileName[ x ] ), T )
  print( paste( "Apply SR to:", inputFileName[ x ]  ) )
  img = antsImageRead( inputFileName[ x ] )
  imgmask = getMask( img )
  cimg = cropImage( img, imgmask )
  if ( !doCrop ) cimg = img
  mdl = createDeepBackProjectionNetworkModel3D( c( dim( cimg ),  1 ),
     numberOfOutputs = 1, numberOfBaseFilters = 64,
     numberOfFeatureFilters = 256, numberOfBackProjectionStages = 7,
     convolutionKernelSize = rep( 3, 3 ),
     strides = rep( 2, 3 ),
     numberOfLossFunctions = 1 )
  load_model_weights_hdf5( mdl, modelFile )
  print("Begin")
  mysr = applySuperResolutionModelPatch( cimg, mdl,
    targetRange = c( 127.5, -127.5 ), lowResolutionPatchSize=24,
    strideLength = strideLength )
  if ( doPerm ) {
    permarr = c( 2, 1, 3 )
    ref0 = as.antsImage( aperm( as.array( cimg ), permarr ) )
    mysr0 = applySuperResolutionModelPatch( ref0, mdl,
      targetRange = c( 127.5, -127.5 ), lowResolutionPatchSize=24,
      strideLength = strideLength )
    mysr0 = as.antsImage( aperm( as.array( mysr0 ),  permarr ) )
    mysr0 = antsCopyImageInfo( mysr, mysr0 )
    mysr = mysr0 * 0.5 + mysr
    }
  dir.create( file.path( './', 'results_GPU'), showWarnings = FALSE)
  antsImageWrite( mysr, paste0( './results_GPU/', outfn, "_strides", strideLength, "_SR.nii.gz" ) )
}

