import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from utils import *
import numpy as np
import gym
from copy import deepcopy
from network import ActorCritic
from train import run_training_procedure
from utils import process_screen

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
	argparse.add_argument('--learning_rate', type = float)
	argparse.add_argument('--environment', type = str)
	argparse.add_argument('--optimizer', type = str)
	argparse.add_argument('--gamma', type = float)
	argparse.add_argument('--checkpoint_dir', action = readable_dir)
	argparse.add_argument('--gifs_dir', action = readable_dir)
	argparse.add_argument('--checkpoint_path', action = readable_file)
	argparse.add_argument('--log_dir', action = readable_dir)
	argparse.add_argument('--threads', type = int, default = 8)
	argparse.add_argument('--checkpoint_save_interval', type = int)
	# argparse.add_argument('--target_update_interval', type = int)
	argparse.add_argument('--update_intervals', type = int)
	argparse.add_argument('--gifs_save_interval', type = int)
	argparse.add_argument('--gradient_clipping', type = float)

	args = argparse.parse_args()

	print('Creating {} environments for parallel processing'.format(args.threads))
	args.environments = [gym.make(args.environment) for _ in range(args.threads)]

	args.optimizer = get_optimizer(args.optimizer)(args.learning_rate)
	args.actor_critic = ActorCritic(args.environments[0].action_space.n)
	args.actor_critic.set_threads(args.threads)
	args.target_network = args.actor_critic
	# args.target_network.set_threads(args.threads)

	sample_input = process_screen(args.environments[0].observation_space.sample())
	args.actor_critic(sample_input, 0)
	args.actor_critic.reset_thread_states(0)
	# args.target_network(sample_input, 0)
	# args.target_network.reset_thread_states(0)
	if args.checkpoint_path != None:
		args.actor_critic.load_weights(args.checkpoint_path)
		# args.target_network.load_weights(args.checkpoint_path)

	args.summary_writer = tf.summary.create_file_writer(args.log_dir)


	run_training_procedure(args)
