def generate_syn_name_list():
	image_list = []
	normal_list = []
	albedo_list = []
	mask_list = []
	light_list = []
	folder_count = len(next(os.walk('data/synthetic_data/DATA_pose_15/'))[1])

	# generating list for image, albedo, normal, lighting SH and mask in a similar order
	for i in range(0,folder_count):
	folder = str(i+1).zfill(4)
	print(folder)
	path = 'data/synthetic_data/DATA_pose_15/' + folder + '/'
	addrs = os.walk(path)
	for root, dirs, filename in addrs:
		filename.sort() 
		for file in filename:
			if (file.endswith(".png") and 'face' in file):
				image_list.append(os.path.join(root, file))       
			elif (file.endswith(".png") and 'albedo' in file):
				albedo_list.append(os.path.join(root, file))
			elif (file.endswith(".png") and 'normal' in file):
				normal_list.append(os.path.join(root, file))
			elif (file.endswith(".png") and 'mask' in file):
				mask_list.append(os.path.join(root, file))         
			elif (file.endswith(".txt")):
				light_list.append(os.path.join(root, file))

	# converting list to numpy array 
	image_list = np.asarray(image_list)
	albedo_list = np.asarray(albedo_list)
	normal_list = np.asarray(normal_list)
	light_list = np.asarray(light_list)
	mask_list = np.asarray(mask_list)

	features = np.transpose(np.asarray([image_list, mask_list]))
	labels = np.transpose(np.asarray([normal_list, albedo_list, light_list]))

	return features, labels

def train_validation_split(features, labels):
	# Assigning the index values for train and validation data
	n_total = features.shape[0]
	train_min_index = int(n_total * 0)
	train_max_index = int(n_total * 0.8)
	test_min_index = int((n_total * 0.8))
	test_max_index= int(n_total * 1)

	train_features = features[:train_max_index]
	train_labels = labels[:train_max_index]
	test_features = features[test_min_index:]
	test_labels = labels[test_min_index:]

	return train_features, train_labels, test_features, test_labels

# Input function for generating input pipeline for training and prediction
def input_fn(features, labels=None, perform_shuffle=False, repeat_count=1, batch_size=10):
	def _parse_function(feature, label):
		image = tf.read_file(feature[0])
		mask = tf.read_file(feature[1])
		image = tf.image.decode_png(image, channels=3)
		mask = tf.image.decode_png(mask, channels=3)
		# This will convert to float values in [0, 1]
		image = tf.image.convert_image_dtype(image, tf.float32)
		mask = tf.image.convert_image_dtype(mask, tf.float32)
		image = tf.image.resize_images(image, [128, 128])
		mask = tf.image.resize_images(mask, [128,128])
		# Labels
		normal = tf.read_file(label[0])
		albedo = tf.read_file(label[1])
		normal = tf.image.decode_png(normal, channels=3)
		albedo = tf.image.decode_png(albedo, channels=3)
		# This will convert to float values in [0, 1]
		normal = tf.image.convert_image_dtype(normal, tf.float32)
		albedo = tf.image.convert_image_dtype(albedo, tf.float32)
		# Resize image
		normal = tf.image.resize_images(normal, [128, 128])
		albedo = tf.image.resize_images(albedo, [128,128])
		# Light
		light = tf.read_file(label[2])
		return image, mask, normal, albedo, light

	def _parse_function_label_none(feature, label):
		image = tf.read_file(feature[0])
		mask = tf.read_file(feature[1])
		image = tf.image.decode_jpeg(image, channels=3)
		mask = tf.image.decode_jpeg(mask, channels=3)
		# This will convert to float values in [0, 1]
		image = tf.image.convert_image_dtype(image, tf.float32)
		mask = tf.image.convert_image_dtype(mask, tf.float32)
		image = tf.image.resize_images(image, [128, 128])
		mask = tf.image.resize_images(mask, [128,128])
		return image, mask

	if labels is None:
		labels = [0]*len(features)
		dataset = tf.data.Dataset.from_tensor_slices((features, labels))
		dataset = dataset.shuffle(features.shape[0])
		dataset = dataset.map(_parse_function_label_none, num_parallel_calls=4)
		dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
		dataset = dataset.batch(batch_size)  # Batch size to use
		iterator = dataset.make_one_shot_iterator()
		batch_images, batch_masks = iterator.get_next()
		return [batch_images, batch_masks], [None]
	else:
		dataset = tf.data.Dataset.from_tensor_slices((features, labels))
		dataset = dataset.shuffle(features.shape[0])
		dataset = dataset.map(_parse_function, num_parallel_calls=4)
		dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
		dataset = dataset.batch(batch_size)  # Batch size to use
		iterator = dataset.make_one_shot_iterator()
		batch_images, batch_masks, batch_normals, batch_albedos, batch_lights = iterator.get_next()
		return [batch_images, batch_masks], [batch_normals, batch_albedos, batch_lights]

# Model function for Estimator
def model_fn(features, labels, mode, params):
	"""
		The defination of all the blocks to construct the network
		
		Args:
			features: images as dictionary
			labels: albedo, normal and lighting values as dictionary
			mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
			params: a parameter dictionary with the following keys: batch_size,
				learning_rate
		Returns:
			ModelFnOps for Estimator API.
	"""
	
	def _get_input_tensors(features, labels):
		"""Converts the input dict into image, albedo, normal and lighting tensors."""
		image = tf.multiply(features[0], features[1])
		albedo = None
		normal = None
		light = None
		if mode != tf.estimator.ModeKeys.PREDICT:
			albedo = labels[1]
			normal = labels[0]
			light = labels[2]
			function_to_map = lambda x: tf.string_split([x], delimiter='\t').values
			light = tf.map_fn(function_to_map, light)
			light = tf.string_to_number(
				light,
				out_type=tf.float32)
		
		return image, albedo, normal, light
	
	def _encoder_layer(image):
		"""Encoder block"""
		filters = [ 64, 128, 256, 256, 256]
		encoder_op_each_layer = []
		convolved = image
		for i in range(5):
			convolved = tf.layers.conv2d(
				convolved,
				filters = filters[i],
				kernel_size = 4,
				strides = 2,
				padding = "SAME",
				name = "encoder_conv2d_%d" % i)
			if i > 0:
				convlved = tf.layers.batch_normalization(
					convolved,
					training=(mode == tf.estimator.ModeKeys.TRAIN),
					name = "encoder_batch_norm_%d" % i)
			convolved = tf.nn.relu(
				convolved,
				name=None)
			# saving the output from each layer in a list which will be used for skip connection
			encoder_op_each_layer.append(convolved)
		flatten_output = tf.contrib.layers.flatten(convolved)
		dense_output = tf.layers.dense(
			flatten_output,
			256,
			name = "encoder_dense")
		return dense_output, encoder_op_each_layer
	
	def _light_mlp(encoder_output):
		"""Light MLP block"""
		light_mlp_output = tf.layers.dense(
			encoder_output,
			256,
			name = "light_dense_layer1")
		light_mlp_output = tf.layers.dense(
			light_mlp_output,
			27,
			name = "light_dense_layer2")
		return light_mlp_output
	
	def _albedo_mlp(encoder_output):
		"""MLP block for albedo"""
		albedo_mlp_output = tf.layers.dense(
			encoder_output,
			256,
			name = "albedo_mlp_dense_layer")
		albedo_mlp_output = tf.reshape(
			albedo_mlp_output,
			[params.batch_size, 1, 1, 256])
		albedo_mlp_output = tf.keras.layers.UpSampling2D((4, 4))(albedo_mlp_output)
		return albedo_mlp_output
	
	def _normal_mlp(encoder_output):
		"""MLP block for normal"""
		normal_mlp_output = tf.layers.dense(
			encoder_output,
			256,
			name = "normal_mlp_dense_layer"
		)
		normal_mlp_output = tf.reshape(
			normal_mlp_output,
			[params.batch_size, 1, 1, 256])
		normal_mlp_output = tf.keras.layers.UpSampling2D((4, 4))(normal_mlp_output)
		return normal_mlp_output
	
	def _albedo_decoder(albedo_mlp_output, encoder_op_each_layer):
		"""Decoder block for albedo"""
		filters = [ 256, 256, 256, 128, 64]
		deconvolved = albedo_mlp_output

		for i in range(5):
			deconvolved_input = tf.concat(
				[deconvolved, encoder_op_each_layer[5-(i+1)]],
				3)
			deconvolved = tf.layers.conv2d_transpose(
				deconvolved_input,
				filters = filters[i],
				kernel_size = 4,
				strides = 2,
				padding = "SAME",
				name = "albedo_decoder_deconv2d_%d" % i)
			deconvlved = tf.layers.batch_normalization(
				deconvolved,
				training=(mode == tf.estimator.ModeKeys.TRAIN),
				name = "albedo_decoder_batch_norm_%d" % i)
			deconvolved = tf.nn.relu(
				deconvolved,
				name=None)
		albedo_decoder_output = tf.layers.conv2d(
				deconvolved,
				filters = 3,
				kernel_size = 1,
				name = "albedo_decoder_conv2d")
		return albedo_decoder_output
	
	def _normal_decoder(normal_mlp_output, encoder_op_each_layer):
		"""Decoder block for normal"""
		filters = [ 256, 256, 256, 128, 64]
		deconvolved = normal_mlp_output
		for i in range(5):
			deconvolved_input = tf.concat(
					[deconvolved, encoder_op_each_layer[5-(i+1)]],
					3)
			deconvolved = tf.layers.conv2d_transpose(
				deconvolved_input,
				filters = filters[i],
				kernel_size = 4,
				strides = 2,
				padding = "SAME",
				name = "normal_decoder_deconv2d_%d" % i)
			deconvlved = tf.layers.batch_normalization(
				deconvolved,
				training=(mode == tf.estimator.ModeKeys.TRAIN),
				name = "normal_decoder_batch_norm_%d" % i)
			deconvolved = tf.nn.relu(
				deconvolved,
				name=None)
		normal_decoder_output = tf.layers.conv2d(
				deconvolved,
				filters = 3,
				kernel_size = 1,
				name = "normal_decoder_conv2d")
		return normal_decoder_output
	
	# Build the model
	image, albedo, normal, light = _get_input_tensors(features, labels)
	encoder_output, encoder_op_each_layer = _encoder_layer(image)
	light_mlp_output = _light_mlp(encoder_output)
	albedo_mlp_output = _albedo_mlp(encoder_output)
	normal_mlp_output = _normal_mlp(encoder_output)
	albedo_decoder_output = _albedo_decoder(albedo_mlp_output, encoder_op_each_layer)
	normal_decoder_output = _normal_decoder(normal_mlp_output, encoder_op_each_layer)
	#returning estimator if the modoe is predict
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions={
					"image": image,
					"mask": features[1],
					"normal": normal_decoder_output,
					"albedo": albedo_decoder_output,
					"light": light_mlp_output})
	else:
		# Add the loss
		normal_loss = tf.reduce_mean(
			tf.losses.absolute_difference(normal, normal_decoder_output))
		albedo_loss = tf.reduce_mean(
			tf.losses.absolute_difference(albedo, albedo_decoder_output))
		lighting_loss = tf.losses.mean_squared_error(light, light_mlp_output)
		final_loss = tf.reduce_mean(0.5*normal_loss + 0.5*albedo_loss + 0.1*lighting_loss)
		# Add the optimizer
		train_op = tf.contrib.layers.optimize_loss(
			loss=final_loss,
			global_step=tf.train.get_global_step(),
			learning_rate=params.learning_rate,
			optimizer="Adam")

		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions={
					"normal": normal_decoder_output,
					"albedo": albedo_decoder_output,
					"light": light_mlp_output},
			loss=final_loss,
			train_op=train_op,
			eval_metric_ops={
					"albedo_loss": tf.metrics.mean_squared_error(albedo, albedo_decoder_output),
					"normal_loss": tf.metrics.mean_squared_error(normal, normal_decoder_output),
					"lighting_loss": tf.metrics.mean_squared_error(light, light_mlp_output)})

def create_estimator(run_config, batch_size, learning_rate):
	"""Creates an Experiment configuration based on the estimator and input fn."""
	model_params = tf.contrib.training.HParams(
		batch_size = batch_size,
		learning_rate = learning_rate)

	estimator = tf.estimator.Estimator(
		model_fn=model_fn,
		config=run_config,
		params=model_params)

	return estimator
	
def create_specs(batch_size, epochs):
	train_spec = tf.estimator.TrainSpec(
		input_fn=lambda: input_fn(
			train_features,
			labels=train_labels,
			perform_shuffle=True,
			repeat_count=epochs,
			batch_size=batch_size))
	
	eval_spec = tf.estimator.EvalSpec(
		input_fn=lambda: input_fn(
			test_features,
			labels=test_labels,
			perform_shuffle=False,
			batch_size=batch_size),
		steps=10,
		name='validation',
		start_delay_secs=150,
		throttle_secs=200)

	return train_spec, eval_spec