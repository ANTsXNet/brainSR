library( ANTsRNet )
library( tensorflow )
library( keras )
args <- commandArgs( trailingOnly = TRUE )

if ( length( args ) < 2 )
  {
  m1 = "Usage: \n Rscript applySR_GPU.R inputFiles modelFile \n"
  m2 = "   optional-stride-length optional-do-crop optional-do-perm\n"
  m3 = "   optional-do-translation\n"
  m4 = " NOTE: do-perm is an integer and sets the number of array permutations\n"
  m5 = " NOTE: do-translation is an integer and sets the number of translations\n"
  m6 = " NOTE: translation and permutation => ensembling\n"
  helpMessage <- paste( m1, m2, m3, m4, m5, m6 )
  stop( helpMessage )
  } else {
  inputFileName <- Sys.glob( args[1] )
  modelFile <- args[2]
  }
strideLength = 14
doCrop = 0
doPerm = 0
doTrans = 0
if ( length( args ) > 2 ) strideLength = as.numeric( as.character( args[3] ) )
if ( length( args ) > 3 ) doCrop = as.numeric( as.character( args[4] ) )
if ( length( args ) > 4 ) doPerm = as.numeric( as.character( args[5] ) )
if ( length( args ) > 5 ) doTrans = as.numeric( as.character( args[6] ) )
print( paste( "Options= Crop:", doCrop, "doPerm: ", doPerm, "doTrans: ", doTrans) )
if ( doPerm > 0 ) library(PerMallows)
# mdl = load_model_hdf5( modelFile )
for ( x in 1:length( inputFileName ) ) {
  outfn = tools::file_path_sans_ext( basename( inputFileName[ x ] ), T )
  print( paste( "Apply SR to:", inputFileName[ x ]  ) )
  img = antsImageRead( inputFileName[ x ] )
  if ( doCrop > 0 ) {
    imgmask = getMask( img )
    cimg = cropImage( img, imgmask )
  } else cimg = img
  mdl = createDeepBackProjectionNetworkModel3D( c( 24, 24, 24,  1 ),
     numberOfOutputs = 1, numberOfBaseFilters = 64,
     numberOfFeatureFilters = 256, numberOfBackProjectionStages = 7,
     convolutionKernelSize = rep( 3, 3 ),
     strides = rep( 2, 3 ),
     numberOfLossFunctions = 1 )
  load_model_weights_hdf5( mdl, modelFile )
  
  print("Begin")
  iarr = list()
  if ( doTrans == 0 | doPerm > 0 ) {
    mysr = applySuperResolutionModelPatch( cimg, mdl,
      targetRange = c( 127.5, -127.5 ), lowResolutionPatchSize=24,
      strideLength = strideLength )
    iarr[[ 1 ]] = mysr
    }
  if ( doPerm > 0 ) {
    for ( dp in 1:doPerm ) {
      permarr = oarr = 1:3
      cat( paste0("...p-",dp,"") )
      while (  all( permarr == oarr ) ) permarr = sample( oarr )
      ref0 = as.antsImage( aperm( as.array( cimg ), permarr ) )
      mysr0 = applySuperResolutionModelPatch( ref0, mdl,
        targetRange = c( 127.5, -127.5 ), lowResolutionPatchSize=24,
        strideLength = strideLength )
      mysr0 = as.antsImage( aperm( as.array( mysr0 ),  inverse.perm(permarr) ) )
      mysr0 = antsCopyImageInfo( mysr, mysr0 )
      iarr[[ length(iarr) + 1 ]] = mysr0
      }
    }
  if ( doTrans > 0 ) {
    augger <- function( image, trans, interpolation = 'linear' ) {
      moveX1 = createAntsrTransform(dimension=3, type="Euler3DTransform", translation=trans )
      shift1 = applyAntsrTransform(transform=moveX1, data=image, reference=image, interpolation=interpolation )
      srout = applySuperResolutionModelPatch( shift1, mdl,
        targetRange = c( 127.5, -127.5 ), lowResolutionPatchSize=24,
        strideLength = strideLength )
      return( applyAntsrTransform( transform = invertAntsrTransform( moveX1 ), data=srout, reference=srout, interpolation=interpolation  ) )
      }
    for ( dp in 1:doTrans ) {
      scl = 0.5
      trns = antsGetSpacing( cimg ) * rnorm(3) * scl
      cat( paste0("...aug-",dp,"-t=/\n", paste0( trns, collapse='x' ), '\n/x<o>x' ) )
      iarr[[ length(iarr) + 1 ]] = augger( cimg, trns )
      }
    }
#  bilin = resampleImageToTarget( cimg, iarr[[1]] )
  for ( k in 1:length( iarr ) ) {
#    iarr[[k]] = linMatchIntensity( iarr[[k]], bilin, polyOrder = 2, truncate=FALSE )
    iarr[[k]] = iarr[[k]] * 0.5 + iMath( iarr[[k]], "Sharpen" ) * 0.5
    }
  if ( length( iarr ) > 0 ) {
    cat( "...ensembling and blending...")
    wt = 0.25
    avg = antsAverageImages( iarr )
    mysr = avg * wt + iMath( avg, "Sharpen" ) * ( 1 - wt )
  }
  dir.create( file.path( './', 'results_GPU'), showWarnings = FALSE)
  antsImageWrite( mysr, paste0( './results_GPU/', outfn, "_strides", strideLength, "_SR.nii.gz" ) )
}
