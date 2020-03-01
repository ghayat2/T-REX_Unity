'''
Reacher network (also provided in ml-agents/mlagents/trainers/ppo/custom_reward/network_reacher.py)
'''

import tensorflow as tf
from utils.ops import conv2d, flatten, dense, lrelu


class RewardNetReacher:
    def __init__(self, dense_size=16, lrelu_alpha=0.01,
                 scope='network'):

        self.dense_size = dense_size
        self.lrelu_alpha = lrelu_alpha
        self.scope = scope

    def forward_pass(self, state_in, reshape=True, sigmoid_out=False, reuse=None):

        self.state_in = state_in

        shape_in = self.state_in.get_shape().as_list()

        # Get number of input channels for weight/bias init
        channels_in = shape_in[-1]

        with tf.variable_scope(self.scope, reuse=reuse):

            if reshape:
                # Reshape [batch_size, traj_len, H, W, C] into [batch_size*traj_len, H, W, C]
                self.state_in = tf.reshape(self.state_in, [-1, shape_in[1], shape_in[3], shape_in[2]])

            self.layer1 = dense(self.state_in, 16, scope='layer1')

            self.layer2 = dense(self.layer1, 8, scope='layer2')

            self.layer3 = dense(self.layer2, 1, scope='layer3')

            if reshape:
                # Reshape [batch_size, traj_len, H, W, C] into [batch_size*traj_len, H, W, C]
                shape_in = self.layer3.get_shape().as_list()
                self.layer3 = tf.reshape(self.layer3, [-1, shape_in[1], shape_in[3], shape_in[2]])

                self.output = dense(self.layer3, 1, scope='output')

            if sigmoid_out:
                self.output = tf.nn.sigmoid(self.output)

            if reshape:
                # Reshape 1d reward output [batch_size*traj_len] into batches [batch_size, traj_len]
                self.output = tf.reshape(self.output, [-1, shape_in[1]])

            self.network_params = tf.trainable_variables(scope=self.scope)

        return self.output

    def create_train_step(self, high_traj_reward, low_traj_reward, batch_size, optimizer, reduction='mean'):

        # Get cumulative rewards (sum of individual state rewards) for each sample in the batch
        self.high_traj_reward = high_traj_reward
        self.low_traj_reward = low_traj_reward


        self.high_traj_reward_sum = tf.reduce_sum(high_traj_reward, axis=1)

        self.low_traj_reward_sum = tf.reduce_sum(low_traj_reward, axis=1)

        self.logits = tf.concat((tf.expand_dims(self.high_traj_reward_sum, axis=1), tf.expand_dims(self.low_traj_reward_sum, axis=1)),
                           axis=1)
        self.labels = tf.one_hot(indices=[0] * batch_size, depth=2)  # One hot index corresponds to the high reward
        # trajectory (index 0)

        if reduction == 'sum':
            self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits))
        elif reduction == 'mean':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits))
        else:
            raise Exception("Please supply a valid reduction method")
        self.gradients = tf.gradients(self.loss, self.state_in)

        #         # Note - tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits) is equivalent to:
        #         # -1*tf.log(tf.divide(tf.exp(high_traj_reward_sum), (tf.exp(low_traj_reward_sum) + tf.exp(high_traj_reward_sum))))

        self.train_step = optimizer.minimize(self.loss)
