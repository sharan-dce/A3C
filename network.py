import tensorflow as tf
from tensorflow.keras import layers
class ActorCritic(tf.keras.Model):
    def __init__(self, n_actions):
        super(ActorCritic, self).__init__()
        self.n_actions = n_actions

        _layers = []
        _layers.append(layers.Conv2D(filters = 16, kernel_size = 7, strides = 3, activation = tf.nn.leaky_relu))
        _layers.append(layers.Conv2D(filters = 32, kernel_size = 7, strides = 3, activation = tf.nn.leaky_relu))
        _layers.append(layers.Conv2D(filters = 64, kernel_size = 5, strides = 2, activation = tf.nn.leaky_relu))
        _layers.append(layers.Flatten())
        _layers.append(layers.Dense(1024, activation = tf.nn.leaky_relu))

        self._model = tf.keras.models.Sequential(_layers)

        self._actor = layers.Dense(self.n_actions, activation = tf.nn.softmax)
        self._critic = layers.Dense(self.n_actions, activation = None)

    def __call__(self, state):
        output = self._model(state)
        return self._actor(output), self._critic(output)

if __name__ == '__main__':
    pass
