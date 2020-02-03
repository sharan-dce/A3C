import tensorflow as tf
from tensorflow.keras import layers
import types
class ActorCritic(tf.keras.Model):
    def __init__(self, n_actions, use_lstm_layers):
        super(ActorCritic, self).__init__()
        self.n_actions = n_actions
        self.use_lstm_layers = use_lstm_layers

        _layers = []
        if use_lstm_layers == True:
            _layers.append(layers.Conv2D(filters = 16, kernel_size = 7, strides = 3, activation = tf.nn.leaky_relu))
            _layers.append(layers.Conv2D(filters = 32, kernel_size = 7, strides = 3, activation = tf.nn.leaky_relu))
            _layers.append(layers.Lambda(lambda x: tf.expand_dims(x, axis = 1)))
            _layers.append(layers.ConvLSTM2D(filters = 64, kernel_size = 5, strides = 2, activation = tf.nn.tanh, stateful = True))
            _layers.append(layers.Reshape((1, -1)))
            _layers.append(layers.GRU(1024, stateful = True, activation = tf.nn.tanh))

        else:
            _layers.append(layers.Conv2D(filters = 16, kernel_size = 5, strides = 3, activation = tf.nn.leaky_relu))
            _layers.append(layers.Conv2D(filters = 32, kernel_size = 5, strides = 3, activation = tf.nn.leaky_relu))
            _layers.append(layers.Conv2D(filters = 64, kernel_size = 5, strides = 2, activation = tf.nn.leaky_relu))
            _layers.append(layers.Flatten())
            _layers.append(layers.Dense(1024, activation = tf.nn.leaky_relu))

        self._model = tf.keras.models.Sequential(_layers)

        self._actor = layers.Dense(self.n_actions, activation = tf.nn.softmax)
        self._critic = layers.Dense(1, activation = None)

    def __call__(self, state):
        output = self._model(state)
        output_values = self._actor(output), self._critic(output)
        return types.SimpleNamespace(actor = output_values[0], critic = output_values[1])

if __name__ == '__main__':
    pass
