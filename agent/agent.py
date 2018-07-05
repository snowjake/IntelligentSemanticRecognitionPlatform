import tensorflow as tf 
import numpy as np
import logging


class Agent(object):

    def __init__(self, config):
        # 状态空间的维度
        self.state_dim = config['state_dim']
        # 动作空间的维度
        self.action_dim = config['action_dim']
        # 建立神经网络
        self.build_network()
        # init TF session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        check_point = tf.train.get_checkpoint_state('saved_network')
        if check_point and check_point.model_checkpoint_path:
            self.saver.restore(self.session, check_point.model_checkpoint_path)
            logging.info('load model  success')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape=shape)
        return tf.Variable(initial)

    def build_network(self):
        # first layer ,100 units
        # with tf.name_scope('source'):
        with tf.name_scope('source'):
            W1 = self.weight_variable([self.state_dim, 100])
            b1 = self.bias_variable([100])
            W2 = self.weight_variable([100, self.action_dim])
            b2 = self.bias_variable([self.action_dim])

            # input layer
            self.state_input = tf.placeholder('float', [None, self.state_dim])
            # hiden layer
            layer_1 = tf.nn.relu(tf.matmul(self.state_input, W1)+b1)
            # Q value layer
            self.Q_value = tf.matmul(layer_1, W2)+b2

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
            })
        Q_value = Q_value[0]
        return np.argmax(Q_value)


if __name__ == '__main__':
    config = {'state_dim': 4, 'action_dim': 4}
    agent = Agent(config=config)
    print(agent.egreedy_action([0, 1, 1, 1]))




