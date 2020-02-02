import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from utils import *
import numpy as np
import gym


def get_optimizer(optimizer_name):

	if optimizer_name == 'sgd':
		return tf.keras.optimizers.SGD
	elif optimizer_name == 'adam':
		return tf.keras.optimizers.Adam
	elif optimizer_name == 'rms_prop':
		return tf.keras.optimizers.RMSprop
	else:
		print('Unknown optimizer ' + optimizer_name)
		quit()

if __name__ == '__main__':
	from argparse import ArgumentParser
	argparse = ArgumentParser()
	argparse.add_argument('--log_dir', action = readable_dir)
	argparse.add_argument('--learning_rate', type = float)
	argparse.add_argument('--render', action = "store_true")
	argparse.add_argument('--environment', type = str)
	argparse.add_argument('--optimizer', type = str)
	argparse.add_argument('--epochs', type = int)
	argparse.add_argument('--episodes_per_epoch', type = int)
	argparse.add_argument('--tests_per_epoch', type = int)
	argparse.add_argument('--gamma', type = float)
	argparse.add_argument('--checkpoint_dir', action = readable_dir)
	argparse.add_argument('--gifs_dir', action = readable_dir)
	argparse.add_argument('--checkpoint_path', action = readable_file)
	argparse.add_argument('--use_lstm_layers', action = 'store_true')
	argparse.add_argument('--threads', type = int, default = 8)

	args = argparse.parse_args()

	# create environment
	environment = gym.make_environment(args.environment)
    network = 
