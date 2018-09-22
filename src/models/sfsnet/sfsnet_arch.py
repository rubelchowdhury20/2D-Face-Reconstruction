import numpy as np
import tensorflow as tf

def input_fn(data, batch_size = 1, repeat_count = 1):
    # Input function for tf.estimator
    '''      
        Input: train, validation or test data
        
        Output: features and labels as dictionary
    '''
    
    features = np.transpose(data["features"])
    labels = np.transpose(data["labels"])
    
    def parse_function(feature_names=features, label_names=labels):
        
        print(feature_names[0])
        
        # Features
        image_string = tf.read_file(feature_names[0])
        mask_string = tf.read_file(feature_names[1])

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image = tf.image.decode_jpeg(image_string, channels=3)
        mask = tf.image.decode_jpeg(mask_string, channels=3)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        mask = tf.image.convert_image_dtype(mask, tf.float32)
        
        # This will resize the image to required dimensions
        image = tf.image.resize_images(image, [128, 128])
        mask = tf.image.resize_images(mask, [128,128])

        # Labels
        normal_string = tf.read_file(label_names[0])
        albedo_string = tf.read_file(label_names[1])
        light_string = tf.read_file(label_names[2])
        light_string_1 = [light_string]
        nlabel_string = label_names[3]
        alabel_string = label_names[4]
        mlabel_string = label_names[5]
        
        # Don't use tf.image.decode_image, or the output shape will be undefined
        normal = tf.image.decode_jpeg(normal_string, channels=3)
        albedo = tf.image.decode_jpeg(albedo_string, channels=3)
        light = tf.string_split(light_string_1,delimiter='\t').values

        # This will convert to float values in [0, 1]
        normal = tf.image.convert_image_dtype(normal, tf.float32)
        albedo = tf.image.convert_image_dtype(albedo, tf.float32)
        light = tf.string_to_number(light, out_type=tf.float32)
        nlabel = tf.string_to_number(nlabel_string, out_type=tf.int32)
        alabel = tf.string_to_number(alabel_string, out_type=tf.int32)
        mlabel = tf.string_to_number(mlabel_string, out_type=tf.int32)

        normal = tf.image.resize_images(normal, [128, 128])
        albedo = tf.image.resize_images(albedo, [128,128])

        return image, mask, normal, albedo, light, nlabel, alabel, mlabel

    # Convert the inputs to a Dataset.    
    if labels is None:
      
        labels = [0]*len(features)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(parse_function, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)  # Batch size to use
        dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
        dataset = dataset.prefetch(batch_size)
        
        iterator = dataset.make_one_shot_iterator()
        batch_images, batch_masks = iterator.get_next() 
        
        features = {"images": batch_images, "masks": batch_masks}
        labels = None
   
    else:   

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))    

        dataset = dataset.map(parse_function, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size=features.shape[0])
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(repeat_count) # Number of epochs
        dataset = dataset.prefetch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        images, masks, normals, albedos, lights, lnormal, lalbedo, lmask = iterator.get_next()

        features = {"images": images, "masks": masks}
        labels = {"normals": normals, "albedos": albedos, "lightings": lights, "nlabels":lnormal, "alabels":lalbedo, "mlabels":lmask}      

    return features, labels


def model_fn(features, labels, mode, params):
    # The network architecture along with loss functions 
    '''
        Input: Features(i.e. images and corresponding masks)
               Labels(i.e. Normal, Albedo, lighting and their labels)
               Mode(one of tf.estimator.ModeKeys.{TRAIN, PREDICT, EVAL})
               Parameters(A dictionary containing all the hyperparameters for the training
               
        Output:ModelFnOps for Estimator API.
    '''
    def NormLayer(bottom_layer):
        # Normalisation layer(Helper Method)

        sz=bottom_layer.shape
        nor=bottom_layer
        nor=2*nor-1
        
        ssq=tf.norm(nor,axis=3, keepdims=True)
        norm=tf.tile(ssq,[1,1,1,sz[3]]) + 1e-8
        top_layer=tf.divide(nor,norm)

        return top_layer
    
    def ShadingLayer(normal_output, light_output):
        # Shading Layer(Helper Method)
        
        sz = normal_output.shape
        att = np.pi*np.array([1, 2.0/3, 0.25])

        c1=att[0]*(1.0/np.sqrt(4*np.pi))
        c2=att[1]*(np.sqrt(3.0/(4*np.pi)))
        c3=att[2]*0.5*(np.sqrt(5.0/(4*np.pi)))
        c4=att[2]*(3.0*(np.sqrt(5.0/(12*np.pi))))
        c5=att[2]*(3.0*(np.sqrt(5.0/(48*np.pi))))
        
        shading_img_batch = []  # Empty list for storing all the shading images of the batch
        for i in range(params.batch_size):
            nx = normal_output[i,:,:,0]
            ny = normal_output[i,:,:,1]
            nz = normal_output[i,:,:,2]
          
            H1 = c1*tf.ones([sz[1],sz[2]])
            H2 = c2*nz
            H3 = c2*nx
            H4 = c2*ny
            H5 = c3*(2*nz*nz - nx*nx -ny*ny)
            H6 = c4*nx*nz
            H7 = c4*ny*nz
            H8 = c5*(nx*nx - ny*ny)
            H9 = c4*nx*ny
          
            shading_img = []
            for j in range(0,3):
                L = light_output[i,j*9:(j+1)*9]
                shading_output = L[0]*H1+L[1]*H2+L[2]*H3+L[3]*H4+L[4]*H5+L[5]*H6+L[6]*H7+L[7]*H8+L[8]*H9
                shading_img.append(shading_output)

            shading_img = tf.stack(shading_img, axis=0)

            shading_img = tf.transpose(shading_img, [1,2,0])

            shading_img_batch.append(shading_img)
          
        shading_img_batch = tf.stack(shading_img_batch, axis=0)
        
        return shading_img_batch
  
    # Initialisation for training
    image = features["images"]
    mask = features["masks"]
    albedo = None
    normal = None
    light = None
    nlabel = None
    alabel = None
    mlabel = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        albedo = labels['albedos']
        normal = labels['normals']
        light = labels['lightings']
        nlabel = labels['nlabels']
        alabel = labels['alabels']
        mlabel = labels['mlabels']


    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    conv1 = tf.layers.conv2d(inputs=image, filters=64, kernel_size=7, strides=1, padding="SAME", activation=tf.nn.relu,
                     kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="conv1")

    conv1 =tf.layers.batch_normalization(inputs=conv1, training=is_training, name="bn1")

    conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu, 
                     kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "conv2")

    conv2 =tf.layers.batch_normalization(inputs=conv2, training=is_training, name="bn2")

    conv3 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=3, strides=2, padding="SAME", activation=None, 
                     kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="conv3")

	#########################################################################################################

    # Normal Residual block 

    nconv = []
    nconvr = []
    nsum = []
    nbn = []

    conv3_n = tf.nn.relu(conv3, name="nrelu1")
    nbn_iter = tf.layers.batch_normalization(inputs=conv3_n, training=is_training, name="nbn1")


    nconv_iter = tf.layers.conv2d(inputs=nbn_iter, filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu, 
                     kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "nconv1")

    nconv_iter =tf.layers.batch_normalization(inputs=nconv_iter, training=is_training, name="nbn1r")

    nconvr_iter = tf.layers.conv2d(inputs=nconv_iter, filters=128, kernel_size=3, strides=1, padding="SAME", activation=None, 
                     kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "nconv1r")

    nsum_iter = tf.add(conv3_n, nconvr_iter, name="nsum1" )
    
    nbn.append(nbn_iter)
    nconv.append(nconv_iter)
    nconvr.append(nconvr_iter)
    nsum.append(nsum_iter)
    for i in range(2, 5):

        nsum[-1] = tf.nn.relu(nsum[-1], name="nrelu"+str(i))
        nbn_iter = tf.layers.batch_normalization(inputs=nsum[-1], training=is_training, name="nbn"+str(i))


        nconv_iter = tf.layers.conv2d(inputs=nbn_iter, filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="nconv"+str(i))

        nconv_iter =tf.layers.batch_normalization(inputs=nconv_iter, training=is_training, name="nbn"+str(i)+"r")

        nconvr_iter = tf.layers.conv2d(inputs=nconv_iter, filters=128, kernel_size=3, strides=1, padding="SAME", activation=None, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="nconv"+str(i)+"r")

        nsum_iter = tf.add(nconvr_iter, nsum[-1], name="nsum"+str(i))
        
        nbn.append(nbn_iter)
        nconv.append(nconv_iter)
        nconvr.append(nconvr_iter)
        nsum.append(nsum_iter)
        
        
    nsum[-1] = tf.nn.relu(nsum[-1], name="nrelu6r")
    nsum[-1] = tf.layers.batch_normalization(inputs=nsum[-1], training=is_training, name="nbn6r")

    # Normal Deconvolutional Block 
    
    # Unable to implement bilinear upsampling(as kernel initializer)
    nup6 = tf.layers.conv2d_transpose(inputs=nsum[-1], filters=128,  kernel_size=4, strides=2, padding="SAME", use_bias=False, name="nup6")

    nconv6 = tf.layers.conv2d(inputs=nup6, filters=128, kernel_size=1, strides=1, padding="VALID", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "nconv6")

    nconv6 = tf.layers.batch_normalization(inputs=nconv6, training=is_training, name="nbn6")

    nconv7 = tf.layers.conv2d(inputs=nconv6, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="nconv7")

    nconv7 = tf.layers.batch_normalization(inputs=nconv7, training=is_training, name="nbn7")


    Nconv0 = tf.layers.conv2d(inputs=nconv7, filters=3, kernel_size=1, strides=1, padding="VALID", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="Nconv0")



    #########################################################################################################

    # Albedo Residual block
    aconv = []
    aconvr = []
    asum = []
    abn = []
    
    
    conv3_a = tf.nn.relu(conv3, name="arelu1")
    abn_iter = tf.layers.batch_normalization(inputs=conv3_a, training=is_training, name="abn1")


    aconv_iter = tf.layers.conv2d(inputs=abn_iter, filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu, 
                     kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "aconv1")

    aconv_iter =tf.layers.batch_normalization(inputs=aconv_iter, training=is_training, name="abn1r")

    aconvr_iter = tf.layers.conv2d(inputs=aconv_iter, filters=128, kernel_size=3, strides=1, padding="SAME", activation=None, 
                     kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "aconv1r")

    asum_iter = tf.add(conv3_a, aconvr_iter, name="asum1" )
    
    abn.append(abn_iter)
    aconv.append(aconv_iter)
    aconvr.append(aconvr_iter)
    asum.append(asum_iter)


    for i in range(2, 5):

        asum[-1] = tf.nn.relu(asum[-1], name="arelu"+str(i))
        abn_iter = tf.layers.batch_normalization(inputs=asum[-1], training=is_training, name="abn"+str(i))


        aconv_iter = tf.layers.conv2d(inputs=abn_iter, filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "aconv"+str(i))

        aconv_iter =tf.layers.batch_normalization(inputs=aconv_iter, training=is_training, name="abn"+str(i)+"r")

        aconvr_iter = tf.layers.conv2d(inputs=aconv_iter, filters=128, kernel_size=3, strides=1, padding="SAME", activation=None, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "aconv"+str(i)+"r")

        asum_iter = tf.add(aconvr_iter, asum[-1], name="asum"+str(i))
        
        abn.append(abn_iter)
        aconv.append(aconv_iter)
        aconvr.append(aconvr_iter)
        asum.append(asum_iter)        

    asum[-1] = tf.nn.relu(asum[-1], name="arelu6r")
    asum[-1] = tf.layers.batch_normalization(inputs=asum[-1], training=is_training, name="abn6r")

    # Albedo Deconvolutional Block 

    aup6 = tf.layers.conv2d_transpose(inputs=asum[-1], filters=128,  kernel_size=4, strides=2, padding="SAME", use_bias=False, name="aup6")

    aconv6 = tf.layers.conv2d(inputs=aup6, filters=128, kernel_size=1, strides=1, padding="VALID", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name = "aconv6")

    aconv6 = tf.layers.batch_normalization(inputs=aconv6, training=is_training, name="abn6")

    aconv7 = tf.layers.conv2d(inputs=aconv6, filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="aconv7")

    aconv7 = tf.layers.batch_normalization(inputs=aconv7, training=is_training, name="abn7")


    Aconv0 = tf.layers.conv2d(inputs=aconv7, filters=3, kernel_size=1, strides=1, padding="VALID", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="Aconv0")


    #########################################################################################################

    # Light Estimation Block

    lconcat1 = tf.concat([nsum[-1], asum[-1]], axis=3, name="lconcat1")
    lconcat2 = tf.concat([lconcat1, conv3], axis=3, name="lconcat2")
    lconv1 = tf.layers.conv2d(inputs=lconcat2, filters=128, kernel_size=1, strides=1, padding="VALID", activation=tf.nn.relu, 
                        kernel_initializer=tf.contrib.layers.xavier_initializer(), trainable=is_training, name="lconv1")

    lconv1 = tf.layers.batch_normalization(inputs=lconv1, training=is_training, name="lbn1")
 
    lpool2r = tf.layers.average_pooling2d(inputs=lconv1, pool_size=64, strides=1, padding="VALID", name="lpool2r")
    
    # Additional flatten layer as compared to Caffe to convert it into a single dimensional vector so that dense operation can be run on it.
    lflat2r = tf.layers.flatten(inputs=lpool2r, name="lflat2r")

    # TODO: How to set std value for the kernel_initializer in the dense layer itself. 
    fc_light = tf.layers.dense(inputs=lflat2r, units=27, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.ones_initializer(), trainable=is_training, name="fc_light")
    
    #########################################################################################################
    # Applying masks and other functions for loss calculations

    #print("Before norm layer")
    recnormal = NormLayer(Nconv0)
    #print("Post norm layer")
    if mode != tf.estimator.ModeKeys.PREDICT:
        normal_m = tf.multiply(labels['normals'], features['masks'], name="mask_norgt")
        albedo_m = tf.multiply(labels['albedos'], features['masks'], name="mask_algt")
    
    arec = tf.multiply(Aconv0, features['masks'], name="mask_al")
    rec = tf.multiply(Nconv0, features['masks'], name="mask_nor")
    
    shading = ShadingLayer(recnormal, fc_light)
    shading = tf.multiply(shading, features['masks'], name='mask_shad')
    #print(Aconv0.shape, shading.shape)
    recon = tf.multiply(Aconv0, shading, name="recon")
    recon_mask = tf.multiply(features['masks'], recon, name="mask_recon")
    data_mask = tf.multiply(features['masks'], features['images'], name="mask_data")
    
    #########################################################################################################
    
    # Return the estimator if in PREDICT mode
    
    if mode == tf.estimator.ModeKeys.PREDICT :
        
        predicted_estimator = tf.estimator.EstimatorSpec(mode=mode, predictions={'image':image, 'normal':rec, 'albedo':arec, \
                                                                                 'light':fc_light, 'recon':recon_mask, 'shading': shading})
    
    else:
        
        def l1_loss_layer_wt(input1, input2, input3, param_str):
            # Custom L1 Loss function
            #print(input3)
            if (input3 == 1):
                wt = param_str['wt_real']
            else:
                wt = param_str['wt_syn']
            
            l1_loss = tf.reduce_mean(tf.scalar_mul(wt, tf.losses.absolute_difference(input1, input2)))

            return l1_loss

        def l2_loss_layer_wt(input1, input2, input3, param_str):
            # Custom L2 Loss function
            #print(param_str)
            if input3 == 1:
                wt = param_str['wt_real']
            else:
                wt = param_str['wt_syn']
                	
            
            l2_loss = tf.reduce_mean(tf.scalar_mul(wt, tf.losses.mean_squared_error(input1, input2)))

            return l2_loss
        
        params={'wt_real':0.5, 'wt_syn':0.5}    # MODIFY LATER

        # L2 loss for the lighting coefficients.
        lloss = l2_loss_layer_wt(fc_light, light, labels['mlabels'], params)

        # L1 loss for albedo, normal and reconstruction.

        aloss = l1_loss_layer_wt(arec, albedo_m, labels['alabels'], params)
        reconloss = l1_loss_layer_wt(recon_mask, data_mask, labels['mlabels'], params)
        loss = l1_loss_layer_wt(rec, normal_m, labels['nlabels'], params)   
        
        final_loss = tf.reduce_mean(0.5*lloss + 0.5*aloss + 0.5*reconloss + 0.1*loss)


        # Adding optimizer
        train_op = tf.contrib.layers.optimize_loss(loss=final_loss, global_step=tf.train.get_global_step(), \
                                                   learning_rate=0.00001, optimizer="Adam")
        
        predicted_estimator = tf.estimator.EstimatorSpec(mode=mode, predictions={'normal':rec, 'albedo':arec, 'light':fc_light, \
                                                         'recon':recon_mask, 'shading': shading}, loss=final_loss, train_op=train_op,\
                                                         eval_metric_ops={'normal_loss':tf.metrics.mean_absolute_error(rec, normal_m),\
                                                                          'albedo_loss':tf.metrics.mean_absolute_error(arec, albedo_m),\
                                                                          'lighting_loss':tf.metrics.mean_squared_error(fc_light, light),\
                                                                          'reconstruction_loss':tf.metrics.mean_absolute_error(recon_mask, data_mask)})
        
    return predicted_estimator


def create_estimator_and_specs(run_config):
    # Creates a trial configuration based on the estimator and input fn.
    model_params = tf.contrib.training.HParams(batch_size = sfsnet_batch_size, learning_rate = sfsnet_learning_rate)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=model_params)
    
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_data, batch_size = sfsnet_batch_size, repeat_count = sfsnet_epochs)) # repeat_count = epoch

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(test_data, batch_size = sfsnet_batch_size), steps=10, name='validation', start_delay_secs=150, throttle_secs=500)

    return estimator, train_spec, eval_spec     
            