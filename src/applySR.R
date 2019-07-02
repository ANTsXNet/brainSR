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

# this is unfortunate but b/c of some likely keras bug we need to include the
# model code here ( preventing this from being fully general ) and set the
# weights from the model file
perceptualDBPN3D <- function( inputImageSize,
                               numberOfOutputs = 1,
                               numberOfBaseFilters = 64,
                               numberOfFeatureFilters = 256,
                               numberOfBackProjectionStages = 7,
                               convolutionKernelSize = c( 12, 12, 12 ),
                               strides = c( 8, 8, 8 ),
                               stridesUD = c( 2, 2, 2 ),
                               lastConvolution = c( 9, 9, 9 ),
                               numberOfLossFunctions = 1,
                               interpolationType = 'nearest',
                               featureModel,
                               discModel
                             )
{
    upBlock3DInterp <- function( L, numberOfFilters = 64, kernelSize = c( 12, 12, 12 ),
      strides = c( 2, 2, 2 ), itype = interpolationType  )
      {
        if( TRUE )
          {
          L <- L %>% layer_conv_3d( filters = numberOfFilters, use_bias = TRUE,
            kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
          L <- L %>% layer_activation_parametric_relu(
            alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )
          }
      # Scale up
      H0 <- layer_upsampling_3d( L,
        size = c( 2L, 2L, 2L ) ) %>%
        layer_conv_3d( filters = numberOfFilters, use_bias = TRUE,
          kernel_size = c( 3, 3, 3 ), strides = c( 1, 1, 1 ), padding = 'same' )
      H0 <- H0 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

      # Scale down
      L0 <- H0 %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = kernelSize, strides = strides,
        kernel_initializer = 'glorot_uniform', padding = 'same' )

      L0 <- L0 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )
      # Residual

      E <- layer_subtract( list( L0, L ) )

      # Scale residual up
      H1 <- layer_upsampling_3d( E,
        size = c( 2, 2, 2 ) ) %>%
        layer_conv_3d( filters = numberOfFilters, use_bias = TRUE,
          kernel_size = c( 3, 3, 3 ), strides = c( 1, 1, 1 ), padding = 'same' )
      H1 <- H1 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

      # Output feature map
      upBlock <- layer_add( list( H0, H1 ) )

      return( upBlock )
      }


    #
    upBlock3D <- function( L, numberOfFilters = 64, kernelSize = c( 12, 12, 12 ),
      strides = c( 8, 8, 8 ), includeDenseConvolutionLayer = FALSE )
      {
      if( includeDenseConvolutionLayer )
        {
        L <- L %>% layer_conv_3d( filters = numberOfFilters, use_bias = TRUE,
          kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
        L <- L %>% layer_activation_parametric_relu(
          alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )
        }

      # Scale up
      H0 <- L %>% layer_conv_3d_transpose( filters = numberOfFilters,
        kernel_size = kernelSize, strides = strides,
        kernel_initializer = 'glorot_uniform', padding = 'same' )
      H0 <- H0 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

      # Scale down
      L0 <- H0 %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = kernelSize, strides = strides,
        kernel_initializer = 'glorot_uniform', padding = 'same' )
      L0 <- L0 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

      # Residual
      E <- layer_subtract( list( L0, L ) )

      # Scale residual up
      H1 <- E %>% layer_conv_3d_transpose( filters = numberOfFilters,
        kernel_size = kernelSize, strides = strides,
        kernel_initializer = 'glorot_uniform', padding = 'same' )
      H1 <- H1 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

      # Output feature map
      upBlock <- layer_add( list( H0, H1 ) )

      return( upBlock )
      }

    downBlock3D <- function( H, numberOfFilters = 64, kernelSize = c( 12, 12, 12 ),
      strides = c( 8, 8, 8 ), includeDenseConvolutionLayer = FALSE )
      {
      if( includeDenseConvolutionLayer )
        {
        H <- H %>% layer_conv_3d( filters = numberOfFilters, use_bias = TRUE,
          kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
        H <- H %>% layer_activation_parametric_relu(
          alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )
        }

      # Scale down
      L0 <- H %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = kernelSize, strides = strides,
        kernel_initializer = 'glorot_uniform', padding = 'same' )
      L0 <- L0 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

      # Scale up
      H0 <- L0 %>% layer_conv_3d_transpose( filters = numberOfFilters,
        kernel_size = kernelSize, strides = strides,
        kernel_initializer = 'glorot_uniform', padding = 'same' )
      H0 <- H0 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

      # Residual
      E <- layer_subtract( list( H0, H ) )

      # Scale residual down
      L1 <- E %>% layer_conv_3d( filters = numberOfFilters,
        kernel_size = kernelSize, strides = strides,
        kernel_initializer = 'glorot_uniform', padding = 'same' )
      L1 <- L1 %>% layer_activation_parametric_relu(
        alpha_initializer = 'zero', shared_axes = c( 1, 2, 3 ) )

      # Output feature map
      downBlock <- layer_add( list( L0, L1 ) )

      return( downBlock )
      }

    inputs <- layer_input( shape = inputImageSize )

    # Initial feature extraction
    model <- inputs %>% layer_conv_3d( filters = numberOfFeatureFilters,
      kernel_size = c( 3, 3, 3 ), strides = c( 1, 1, 1 ), padding = 'same',
      kernel_initializer = "glorot_uniform" )
    model <- model %>% layer_activation_parametric_relu( alpha_initializer = 'zero',
      shared_axes = c( 1, 2, 3 ) )

    # Feature smashing
    model <- model %>% layer_conv_3d( filters = numberOfBaseFilters,
      kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same',
      kernel_initializer = "glorot_uniform" )
    model <- model %>% layer_activation_parametric_relu( alpha_initializer = 'zero',
      shared_axes = c( 1, 2, 3 ) )

    # Back projection
    upProjectionBlocks <- list()
    downProjectionBlocks <- list()

    if ( interpolationType == 'conv' )
      model <- upBlock3D( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides )
    if ( interpolationType %in% c('bilinear','nearest') )
      model <- upBlock3DInterp( model, numberOfFilters = numberOfBaseFilters,
        kernelSize = convolutionKernelSize, strides = strides, itype = interpolationType )


    upProjectionBlocks[[1]] <- model

    for( i in seq_len( numberOfBackProjectionStages ) )
      {
      if( i == 1 )
        {
        model <- downBlock3D( model, numberOfFilters = numberOfBaseFilters,
          kernelSize = convolutionKernelSize, strides = strides )
        downProjectionBlocks[[i]] <- model

        if ( interpolationType == 'conv' )
          model <- upBlock3D( model, numberOfFilters = numberOfBaseFilters,
            kernelSize = convolutionKernelSize, strides = strides )
        if ( interpolationType %in% c('bilinear','nearest') )
          model <- upBlock3DInterp( model, numberOfFilters = numberOfBaseFilters,
            kernelSize = convolutionKernelSize, strides = strides, itype = interpolationType )
        upProjectionBlocks[[i+1]] <- model
        model <- layer_concatenate( upProjectionBlocks )
        } else {
        model <- downBlock3D( model, numberOfFilters = numberOfBaseFilters,
          kernelSize = convolutionKernelSize, strides = strides,
          includeDenseConvolutionLayer = TRUE )
        downProjectionBlocks[[i]] <- model
        model <- layer_concatenate( downProjectionBlocks )

        if ( interpolationType == 'conv' )
          model <- upBlock3D( model, numberOfFilters = numberOfBaseFilters,
            kernelSize = convolutionKernelSize, strides = strides, includeDenseConvolutionLayer = TRUE )
        if ( interpolationType %in% c('bilinear','nearest') )
          model <- upBlock3DInterp( model, numberOfFilters = numberOfBaseFilters,
            kernelSize = convolutionKernelSize, strides = strides, itype = interpolationType )

        upProjectionBlocks[[i+1]] <- model
        model <- layer_concatenate( upProjectionBlocks )
        }
      }

    # Final convolution layer
    outputs <- model %>% layer_conv_3d( filters = numberOfOutputs,
      kernel_size = lastConvolution, strides = c( 1, 1, 1 ), padding = 'same',
      kernel_initializer = "glorot_uniform" )

  outputs2 <- model %>% layer_conv_3d( filters = numberOfOutputs,
      kernel_size = lastConvolution, strides = c( 1, 1, 1 ), padding = 'same',
      kernel_initializer = "glorot_uniform" )

  if ( numberOfLossFunctions == 1 ) {
    deepBackProjectionNetworkModel <- keras_model(
      inputs = inputs,
      outputs = outputs )
    } else {
      olist = list()
      for ( k in 1:numberOfLossFunctions ) olist[[ k ]] = outputs
      deepBackProjectionNetworkModel <- keras_model(
        inputs = inputs,
        outputs = olist )
    }

  myouts = list( outputs )
  if ( ! missing( featureModel ) ) {
    features <- deepBackProjectionNetworkModel$output %>%
      featureModel
    myouts = list( features, outputs )
    }
  if ( ! missing( discModel ) ) {
   discer <-  deepBackProjectionNetworkModel$output %>% discModel
   myouts = lappend( myouts, discer )
   }

  # this is the model we will train
  model <- keras_model(
    inputs = deepBackProjectionNetworkModel$input,
    outputs = myouts
    )
  return( model )
}



mdl = load_model_hdf5( modelFile )
for ( x in 1:length( inputFileName ) ) {
  outfn = tools::file_path_sans_ext( basename( inputFileName[ x ] ), T )
  outfn = ps
  print( outfn )
  img = antsImageRead( inputFileName[ x ] )
  woo = perceptualDBPN3D( c( dim(img),  1 ),
     numberOfOutputs = 1, numberOfBaseFilters = 64,
     numberOfFeatureFilters = 256, numberOfBackProjectionStages = 5,
     convolutionKernelSize = rep( 3, img@dimension),
     strides = rep( 2, img@dimension ),
     numberOfLossFunctions = 1, interpolationType = 'nearest' )
  set_weights( woo, get_weights( mdl ) )
  mysr = applySuperResolutionModel( img, woo )
  dir.create( file.path( './', 'results'), showWarnings = FALSE)
  antsImageWrite( mysr, paste0( './results/', outfn, "_SR.nii.gz" ) )
}
