"""Module defining the neural networks

The input of a deep Q Network is a concatenation of an action and a state.
The output of a deep Q Network is a single scalar, the value of the Q function.

This is unlike conventional deep Q networks, because actions take continuous
values.

There are three usable classes:

    SingleDeepQNetwork: single neural network for all steps
                        state: concatenation of 
                               [action from prev. step, one-hot encoded step #]
                        action: 2*n_sites + 1 dimensional vector,
                                j, hx and hz gates for one step 
                        note: the state is an approximation as it does not
                        contain all the information about the wave function.
                        (only the most recent step is considered)

    InterStepMultiDQN: several neural networks, one for each step

                       for NN at step i:
                       state_i: all actions up to step i-1 (concatenated)
                       action_i: 2*n_sites + 1 dimensional vector, for the gates
                               at step i

    IntraStepMultiDQN: several neural networks, three for each step
                       (one for the j gate, hx gates, and hz gates)
                       
                       state: all actions up to current action (concatenated)
                       action: dimension 1 if j gate,
                               dimension n_sites if hx or hz gate


They are all built on the class StateActionNeuralNetwork.
"""
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow.keras.backend as K
from tdqc.numerics.deep_q_learning.optimizers import AdamOptimizer #, NAGOptimizer


print('Tensorflow verion: ', tf.__version__)
print('Tensorflow file: ', tf.__file__)
print('Keras verion: ', keras.__version__)
print('Keras file: ', keras.__file__)


class StateActionNeuralNetwork():

    def __init__(self,
                 state_dim,
                 action_dim,
                 architecture,
                 sess,
                 max_q_optimizer):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.architecture = architecture
        self.build_model()

        # The K.gradients returns the gradients, that is a gradients tensor, of the first 
        # argument, a scalar tensor to minimize (here, self.model.outputs), w.r.t. the 
        # second argument, a list of variables (here, self.model.inputs). Then, it only 
        # keeps the gradients w.r.t. the variables of the actions (that is the angles). 
        self.output_action_gradient = K.gradients(
            self.model.outputs, self.model.inputs
        )[0][0, -self.action_dim:]
        self.sess = sess

        if max_q_optimizer['algorithm'] == 'NAG':
            self.max_optimizer = NAGOptimizer(
                learning_rate=max_q_optimizer['learning_rate'],
                momentum=max_q_optimizer['momentum'],
                convergence_threshold=max_q_optimizer['convergence_threshold'],
                n_iterations=max_q_optimizer['n_iterations'],
                n_inits=max_q_optimizer['n_initial_actions'],
                initialization=max_q_optimizer['action_initialization']
            )
        elif max_q_optimizer['algorithm'] == 'adam':
            self.max_optimizer = AdamOptimizer(
                alpha=max_q_optimizer['learning_rate'],
                beta_1=max_q_optimizer['beta_1'],
                beta_2=max_q_optimizer['beta_2'],
                epsilon=max_q_optimizer['epsilon'],
                convergence_threshold=max_q_optimizer['convergence_threshold'],
                n_iterations=max_q_optimizer['n_iterations'],
                n_inits=max_q_optimizer['n_initial_actions'],
                initialization=max_q_optimizer['action_initialization']
            )
        else:
            raise ValueError('Unknown max_q_optimizer.')

    def build_model(self):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(self.input_dim,)))
        for n, act in self.architecture:
            self.model.add(Dense(n, activation=act))

    def evaluate(self, gradient, network_input):
        return self.sess.run(gradient,
                             feed_dict={self.model.input: network_input})

    def evaluate_output_action_gradient(self, network_input):
        return self.evaluate(self.output_action_gradient, network_input)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        #  Store history of all the `fit` on the model in:
        #  self.history = {}

    def get_max_output(self, state, optimizer=None):
        if optimizer is None:
            optimizer = self.max_optimizer
        state = np.array(state, dtype=np.float32).flatten()
        if len(state) != self.state_dim:
            raise ValueError(f"len(state)=={len(state)} should be the same as "
                             f"state_dim=={self.state_dim}.")

        def evaluate_output(action):
            # action should be np.float32?
            network_input = np.concatenate([state, action]).reshape(1, -1)
            return self.model.predict(network_input)[0][0]

        def evaluate_gradient(action):
            network_input = np.concatenate([state, action]).reshape(1, -1)
            return self.evaluate_output_action_gradient(network_input)

        return optimizer.run(self.action_dim,
                             evaluate_output,
                             evaluate_gradient)


# class MultiDeepQNetwork():
#     def __init__(self,
#                  tf_seed,
#                  max_q_optimizer,
#                  **other_params
#                  ):
#         tf.random.set_seed(tf_seed)
#         self.sess = tf.compat.v1.keras.backend.get_session()
#         self.max_q_optimizer = max_q_optimizer

#     def build_networks(self):
#         raise NotImplementedError

#     def update_target(self):
#         for network, target_network in zip(self.networks,
#                                            self.target_networks):
#             # use np.array?
#             target_network.set_weights(network.get_weights())

#     def compile(self, optimizer, loss, metrics):
#         for network in self.networks:
#             network.compile(optimizer=optimizer, loss=loss, metrics=metrics)

#     def get_max_output(self, step, state, use_target):
#         raise NotImplementedError

#     def fit(self, action_sequences, ys, batch_size, epochs):
#         raise NotImplementedError


# class InterStepMultiDQN(MultiDeepQNetwork):
#     def __init__(self,
#                  n_steps,
#                  action_dim,
#                  architectures,
#                  **other_params
#                  ):

#         super().__init__(**other_params)
#         self.n_networks = n_steps
#         self.action_dim = action_dim
#         if len(architectures) == 1:
#             self.architectures = architectures * self.n_networks
#         elif len(architectures) == self.n_networks:
#             self.architectures = architectures
#         else:
#             raise ValueError()
#         self.build_networks()
#         for target_network in self.target_networks:
#             target_network.model.summary()

#     def build_networks(self):
#         self.networks = []
#         self.target_networks = []
#         for n, architecture in enumerate(self.architectures):
#             network = StateActionNeuralNetwork(
#                 state_dim=n*self.action_dim,
#                 action_dim=self.action_dim,
#                 architecture=architecture,
#                 sess=self.sess,
#                 max_q_optimizer=self.max_q_optimizer
#             )
#             target_network = StateActionNeuralNetwork(
#                 state_dim=n*self.action_dim,
#                 action_dim=self.action_dim,
#                 architecture=architecture,
#                 sess=self.sess,
#                 max_q_optimizer=self.max_q_optimizer
#             )
#             self.networks.append(network)
#             self.target_networks.append(target_network)

#     def get_max_output(self, step, state, use_target):
#         if use_target:
#             network = self.target_networks[step]
#         else:
#             network = self.networks[step]
#         return network.get_max_output(state)

#     def fit(self, action_sequences, ys, batch_size, epochs):
#         for n in range(self.n_networks):
#             #  state: [a_0, ..., a_{n-1}], action: a_{n}
#             #  => train = [a_0, ..., a_n].flatten() for each training sample
#             train = action_sequences[:, :n+1, :].reshape(
#                 action_sequences.shape[0], -1
#             )
#             y = ys[:, n]

#             assert len(train) == batch_size, ('training data was '
#                                               'not properly processed')
#             assert len(y) == batch_size, ('training data was '
#                                           'not properly processed')

#             self.networks[n].model.fit(train, y, batch_size=batch_size,
#                                        epochs=epochs, verbose=0)


# class IntraStepMultiDQN(MultiDeepQNetwork):
#     def __init__(self,
#                  n_steps,
#                  action_dims,
#                  architectures,
#                  **other_params
#                  ):

#         super().__init__(**other_params)
#         if not isinstance(action_dims, list):
#             raise TypeError
#         self.action_dims = action_dims
#         self.n_networks = n_steps * len(self.action_dims)

#         if len(architectures) == 1:
#             self.architectures = architectures * self.n_networks
#         elif len(architectures) == self.n_networks:
#             self.architectures = architectures
#         else:
#             raise ValueError()
#         self.build_networks()
#         for target_network in self.target_networks:
#             target_network.model.summary()

#     def build_networks(self):
#         self.networks = []
#         self.target_networks = []
#         state_dim, action_dim = 0, 0
#         for n, archi in enumerate(self.architectures):
#             state_dim += action_dim
#             action_dim = self.action_dims[n % len(self.action_dims)]

#             network = StateActionNeuralNetwork(
#                 state_dim=state_dim,
#                 action_dim=action_dim,
#                 architecture=archi,
#                 sess=self.sess,
#                 max_q_optimizer=self.max_q_optimizer
#             )
#             target_network = StateActionNeuralNetwork(
#                 state_dim=state_dim,
#                 action_dim=action_dim,
#                 architecture=archi,
#                 sess=self.sess,
#                 max_q_optimizer=self.max_q_optimizer
#             )

#             self.networks.append(network)
#             self.target_networks.append(target_network)

#     def get_max_output(self, step, state, use_target):
#         if use_target:
#             networks = self.target_networks
#         else:
#             networks = self.networks
#         n_net = step * len(self.action_dims)

#         state = np.array(state, dtype=np.float32).flatten()
#         state_dim = state.shape[0]
#         state = np.concatenate([state, np.zeros(sum(self.action_dims),
#                                                 dtype=np.float32)])

#         full_step_action = np.zeros(sum(self.action_dims))
#         i_a = 0
#         for i_net, action_dim in enumerate(self.action_dims):
#             a, q = networks[n_net + i_net].get_max_output(state[:state_dim])
#             assert a.shape[0] == self.action_dims[i_net]
#             full_step_action[i_a: i_a + self.action_dims[i_net]] = a
#             state[state_dim: state_dim + a.shape[0]] = a
#             state_dim += a.shape[0]
#             i_a += a.shape[0]
#         # do it in three step, adding action result to next network state
#         return (full_step_action, None)

#     def fit(self, action_sequences, ys, batch_size, epochs):
#         train = action_sequences.reshape(action_sequences.shape[0], -1)
#         # for a given sample i, all ys[i, :] should be equal
#         for n in range(self.n_networks):
#             #  state: [a_0, ..., a_{n-1}], action: a_{n}
#             #  => train = [a_0, ..., a_n].flatten() for each training sample
#             #  train = action_sequences[:, :n+1, :].reshape(
#             #      action_sequences.shape[0], -1
#             #  )
#             y = ys[:, 0]

#             assert len(train) == batch_size, ('training data was '
#                                               'not properly processed')
#             assert len(y) == batch_size, ('training data was '
#                                           'not properly processed')

#             self.networks[n].model.fit(
#                 train[:, :self.networks[n].input_dim],
#                 y,
#                 batch_size=batch_size,
#                 epochs=epochs,
#                 verbose=0
#             )


class SingleDeepQNetwork():
    def __init__(self,
                 n_steps,
                 action_dim,
                 architectures,
                 tf_seed,
                 max_q_optimizer,
                 **other_params
                 ):
        print("tf seed = ", tf_seed)
        tf.random.set_seed(tf_seed)
        self.max_q_optimizer = max_q_optimizer
        self.architecture = architectures[0]
        self.action_dim = action_dim
        self.state_dim = n_steps + action_dim
        self.n_steps = n_steps

        self.sess = tf.compat.v1.keras.backend.get_session()
        self.build_networks()
        self.target_network.model.summary()

    def build_networks(self):
        self.network = StateActionNeuralNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            architecture=self.architecture,
            sess=self.sess,
            max_q_optimizer=self.max_q_optimizer
        )

        self.target_network = StateActionNeuralNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            architecture=self.architecture,
            sess=self.sess,
            max_q_optimizer=self.max_q_optimizer
        )
        self.update_target()

    def update_target(self):
        self.target_network.set_weights(self.network.get_weights())

    def compile(self, optimizer, loss, metrics):
        self.network.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_max_output(self, step, state, use_target):
        """
        Arguments:
            step (int)
            state (list of np.array): all actions until current step (not incl)
            use_target (bool)

        Returns:
            np.array: action at step that maximizes the output of the NN.
        """
        one_hot_step = np.zeros(self.n_steps, dtype=np.float32)
        if step == 0:
            state_truncated = np.zeros(self.action_dim, dtype=np.float32)
        else:
            state_truncated = state[-1]
            assert len(state_truncated) == self.action_dim

        one_hot_step[step] = 1.0
        network_state = np.concatenate([one_hot_step, state_truncated])
        if use_target:
            network = self.target_network
        else:
            network = self.network
        return network.get_max_output(network_state)

    def fit(self, action_sequences, ys, batch_size, epochs):
        # action_sequences.shape == (batch_size, n_steps, action_dim)

        # One episode contains `n_steps` network inputs.
        real_batch_size = self.n_steps * batch_size

        # no TD anymore, just MC method:
        # for a given sample i, all ys[i, :] should be equal

        train = np.zeros(shape=(batch_size,
                                self.n_steps,
                                self.state_dim + self.action_dim),
                         dtype=np.float32)

        for n in range(self.n_steps):
            #  state: [one_hot_step, a_{n-1}], action: a_{n}
            #  => train = [one_hot_step, a_{n-1}, a_n].flatten() 
            #  one_hot_step = np.zeros(self.n_steps, dtype=np.float32)
            #  one_hot_step[n] = 1.0

            #  one-hot encoding
            train[:, n, n] = 1.0
            #  state (a_{n-1}) 
            #  if n == 0, the state is juste defined to be np.zeros
            if n > 0:
                train[:, n, self.n_steps: self.n_steps + self.action_dim] = \
                        action_sequences[:, n-1, :]
            #  action (a_{n}) 
            train[:, n, self.n_steps + self.action_dim:] = \
                    action_sequences[:, n, :]


        train = train.reshape(real_batch_size,
                              self.state_dim + self.action_dim)
        y = ys.reshape(real_batch_size)
        self.network.model.fit(train, y, batch_size=real_batch_size,
                               epochs=epochs, verbose=0)
