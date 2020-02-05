import tensorflow as tf
from tensorflow.keras import layers
import types
import dill

class ActorCritic():
    def __init__(self, n_actions):
        self.n_actions = n_actions

        self._layers = []
        self._layers.append(layers.Conv2D(filters = 16, kernel_size = 8, strides = 4, activation = tf.nn.leaky_relu))
        self._layers.append(layers.Conv2D(filters = 32, kernel_size = 4, strides = 2, activation = tf.nn.leaky_relu))
        self._layers.append(layers.Reshape((1, -1)))

        self._layers.append(layers.GRU(512, return_state = True, return_sequences = True))
        self._layers.append(layers.GRU(128, return_state = True))

        self._actor = layers.Dense(self.n_actions, activation = tf.nn.softmax)
        self._critic = layers.Dense(1, activation = None)

    def _get_reset_state_per_thread(self):
        return [[tf.zeros([1, 2048])], [tf.zeros([1, 1024])]]

    def set_threads(self, threads):
        self.threads = threads
        self.thread_states = [self._get_reset_state_per_thread() for _ in range(threads)]

    def reset_thread_states(self, threads = None):
        if type(threads) == list:
            for thread in threads:
                self.thread_states[thread] = self._get_reset_state_per_thread()
        elif threads == None:
            self.set_threads(self.threads)
        else:
            self.thread_states[threads] = self._get_reset_state_per_thread()

    def __call__(self, state, thread_index):
        output = state
        counter = 0
        for _layer in self._layers:
            if type(_layer) == layers.GRU:
                output, self.thread_states[thread_index][counter] = _layer(output, initial_state = self.thread_states[thread_index][counter])
                counter += 1
            else:
                output = _layer(output)
        return self._actor(output), self._critic(output)

    def save_weights(self, checkpoint_path):
        weight_list = []
        for layer in self._layers + [self._actor, self._critic]:
            weight_list.append(layer.get_weights())

        with open(checkpoint_path, 'wb') as dill_file:
            dill.dump(weight_list, dill_file)

    def load_weights(self, checkpoint_path):
        with open(checkpoint_path, "rb") as dill_file:
            weight_list = dill.load(dill_file)
        for layer, weight in zip(self._layers + [self._actor, self._critic], weight_list):
            layer.set_weights(weight)

    def get_weights(self):
        weight_list = []
        for layer in self._layers + [self._actor, self._critic]:
            weight_list.append(layer.get_weights())
        return weight_list

    def set_weights(self, weights):
        for layer, weight in zip(self._layers + [self._actor, self._critic], weights):
            layer.set_weights(weight)

    @property
    def trainable_variables(self):
        vars = []
        for layer in self._layers + [self._actor, self._critic]:
            vars += layer.trainable_variables
        return vars

if __name__ == '__main__':
    pass
    # import os
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # a = ActorCritic(3)
    # state = tf.random.normal([1, 160, 200, 3])
    # a.set_threads(3)
    # output0 = a(state, 0)[0]
    #
    # import numpy as np
    # a.save_weights('./test_ckpt')
    # a.load_weights('./test_ckpt')
