library( ANTsRNet )
library( tensorflow )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript applySR_GPU.R inputFiles modelFile\n" )
  stop( helpMessage )
  } else {
  inputFileName <- Sys.glob( args[1] )
  modelFile <- args[2]
  }

mdl = load_model_hdf5( modelFile )
for ( x in 1:length( inputFileName ) ) {
  outfn = tools::file_path_sans_ext( basename( inputFileName[ x ] ), T )
  print( paste( "Apply SR to:", inputFileName[ x ]  ) )
  img = antsImageRead( inputFileName[ x ] )
  mysr = applySuperResolutionModelPatch( img, mdl,
    targetRange = c( 127.5, -127.5 ), lowResolutionPatchSize=20,
    strideLength=14 )
  dir.create( file.path( './', 'results_GPU'), showWarnings = FALSE)
  antsImageWrite( mysr, paste0( './results_GPU/', outfn, "_SR.nii.gz" ) )
}
