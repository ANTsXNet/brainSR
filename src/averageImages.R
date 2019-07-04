library( ANTsR )
args <- commandArgs( trailingOnly = TRUE )
if( length( args ) != 2 )
  {
  helpMessage <- paste0( "Usage:  Rscript averageImages.R inputFiles outputFile\n" )
  stop( helpMessage )
  } else {
  inputFileName <- Sys.glob( args[1] )
  modelFile <- args[2]
  }

avg = antsAverageImages( inputFileName )
antsImageWrite( avg, modelFile )

