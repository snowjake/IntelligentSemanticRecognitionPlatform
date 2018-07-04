import tensorflow as tf 
from collections import deque
import numpy as np
import random


GAMMA = 0.9  # discount factor
EPSILON = 0.9
REPLAY_SIZE = 500
BATCH_SIZE = 32
TRAIN_SATART = 100


class Agent(object):

    def __init__(self):

        self.update_step = 1000
        # 设置经验池
        self.replay_buffer = deque()
        # 设置时间步
        self.time_step = 0
        # 设置epsilon
        self.epsilon = EPSILON
        # 状态空间的维度
        self.state_dim = 4
        # 动作空间的维度
        self.action_dim = 4
        # 建立神经网络和训练方法
        self.build_network()
        self.build_target_netword()
        self.train_method()
        # init TF session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        check_point = tf.train.get_checkpoint_state('saved_network')
        if check_point and check_point.model_checkpoint_path:
            self.saver.restore(self.session, check_point.model_checkpoint_path)
            print('load model  success')
        else:
            print('can not find old network weight')

    def build_target_netword(self):
        with tf.name_scope('target'):
            # first layer ,100 units
            W1 = self.weight_variable([self.state_dim, 100])
            b1 = self.bias_variable([100])
            W2 = self.weight_variable([100, self.action_dim])
            b2 = self.bias_variable([self.action_dim])

            # input layer
            # self.state_input = tf.placeholder('float', [None, self.state_dim])
            # hiden layer
            layer_1 = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            # Q value layer
            self.target_Q_value = tf.matmul(layer_1, W2) + b2

    def build_network(self):
        # first layer ,100 units
        with tf.name_scope('source'):
            W1 = self.weight_variable([self.state_dim, 100])
            b1 = self.bias_variable([100])
            W2 = self.weight_variable([100, self.action_dim])
            b2 = self.bias_variable([self.action_dim])

            # input layer
            self.state_input = tf.placeholder('float',[None, self.state_dim])
            # hiden layer
            layer_1 = tf.nn.relu(tf.matmul(self.state_input,W1)+b1)
            # Q value layer
            self.Q_value = tf.matmul(layer_1, W2)+b2

    def train_method(self):
        self.action_input = tf.placeholder('float', [None, self.action_dim])
        self.y_input = tf.placeholder('float', [None])
        self.multiply = tf.multiply(self.Q_value, self.action_input)
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.Q_action = Q_action
        self.cost = tf.reduce_mean(tf.square(self.y_input-Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def train_net(self):
        # get random  sample from replay buffer
        minibach = random.sample(self.replay_buffer, BATCH_SIZE)
        state_bach = [data[0] for data in minibach]
        action_bach = [data[1] for data in minibach]
        reward_bach = [data[2] for data in minibach]
        next_state_bach = [data[3] for data in minibach]

        #calcuate Q
        Y_bach =[]
        next_Q = self.target_Q_value.eval(feed_dict = {self.state_input:next_state_bach})
        # print('next_Q:{}'.format(next_Q))
        for i in range(0, BATCH_SIZE):
            done = minibach[i][4]
            if done:
                Y_bach.append(reward_bach[i])
            else:
                Y_bach.append(reward_bach[i]+GAMMA*np.max(next_Q[i]))
        # _, q_action, multiply, q_value, action_input = self.session.run([self.optimizer, self.Q_action, self.multiply,
        #                                           self.Q_value, self.action_input], feed_dict={
        #     self.y_input: Y_bach,
        #     self.action_input: action_bach,
        #     self.state_input: state_bach
        #     })
        self.optimizer.run(feed_dict={
            self.y_input: Y_bach,
            self.action_input: action_bach,
            self.state_input: state_bach
            })
        # print('q_action:{}, q_action.shape:{}'.format(q_action, q_action.shape))
        # print('multiply:{}, multiply.shape:{}'.format(multiply, multiply.shape))
        # print('q_value:{}, q_value.shape:{}'.format(q_value, q_value.shape))
        # print('action:{}, action.shape:{}'.format(action_input, action_input.shape))

    def print_varibles(self):
        for v_source, v_target in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source'), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')):
            v_target.assign(v_source).eval()
            print('##############################################')
            print('#####################source###################')
            print(self.session.run(v_source))
            print('#####################target###################')
            print(self.session.run(v_target))
            print('##############################################')
            # print('varible: ',self.session.run(v_source))

    def precive(self, state, action, reward, state_, done):
        self.time_step += 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] =1
        self.replay_buffer.append((state,one_hot_action,reward,state_,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > TRAIN_SATART:  # after 100 step ,pre  train
            self.train_net()
        if self.time_step % self.update_step == 0:
            self.update_target()

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
            })
        Q_value = Q_value[0]

        if self.epsilon <= 0.1:
            epsilon_rate = 1
        else:
            epsilon_rate = 0.95
        if self.time_step > 200:
            self.epsilon = epsilon_rate*self.epsilon

        if random.random() <= self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(Q_value)

    def update_target(self):
        for v_source, v_target in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='source'), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')):
            v_target.assign(v_source).eval()


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial  = tf.constant(0.01,shape=shape)
        return tf.Variable(initial)

    def save_model(self, step):
        saver = self.saver
        saver.save(self.session, 'saved_network/'+'network' + '-dqn',global_step=step)

    def getLoss(self):
        pass






