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

mdl = load_model_hdf5( modelFile )
for ( x in 1:length( inputFileName ) ) {
  outfn = tools::file_path_sans_ext( basename( inputFileName[ x ] ), T )
  print( paste( "Apply SR to:", inputFileName[ x ]  ) )
  img = antsImageRead( inputFileName[ x ] )
  mysr = applySuperResolutionModel( img, mdl, targetRange = c( 127.5, -127.5) )
  dir.create( file.path( './', 'results'), showWarnings = FALSE)
  antsImageWrite( mysr, paste0( './results/', outfn, "_SR.nii.gz" ) )
}
