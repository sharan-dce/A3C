import argparse
import os
import tensorflow as tf
class readable_dir(argparse.Action):
	def __call__(self, parser, namespace, values, option_string = None):
		prospective_dir = values
		if not os.path.isdir(prospective_dir):
			raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
		if os.access(prospective_dir, os.R_OK):
			setattr(namespace,self.dest,prospective_dir)
		else:
			raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

class readable_file(argparse.Action):
	def __call__(self, parser, namespace, values, option_string = None):
		prospective_file = values
		if not os.path.isfile(prospective_file):
			raise argparse.ArgumentTypeError("file:{0} is not a valid file".format(prospective_file))
		setattr(namespace,self.dest,prospective_file)

def process_screen(image):
	image = tf.expand_dims(tf.convert_to_tensor(image, dtype = tf.float32), axis = 0)
	image = tf.image.resize(image, [84, 84])
	image /= 1e-6 + tf.reduce_max(image)
	return image * 2.0 - 1.0
