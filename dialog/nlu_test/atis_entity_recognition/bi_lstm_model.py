import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from nlu_test.atis_entity_recognition.data_pro.data_parse import Data_Inter
from nlu_test.atis_entity_recognition.utils import check_multi_path


class Bi_LSTM_Crf(object):
    def __init__(self, param_config,
                 embeddings,
                 tag2label,
                 vocab,
                 model_save_path,
                 intent2id=None,
                 label2tag=None,
                 id2intent=None):
        self.batch_size = param_config.get_batch_size
        self.epoch_num = param_config.get_epoch
        self.hidden_dim = param_config.get_hidden_dim
        self.embeddings = embeddings
        self.CRF = param_config.get_crf
        self.update_embedding = param_config.get_update_embedding
        self.dropout_keep_prob = param_config.get_dropout
        self.optimizer = param_config.get_optimizer
        self.lr = param_config.get_lr
        self.clip_grad = param_config.get_clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        print('self.num_tags:', self.num_tags)
        print('self.tag2label:', self.tag2label)
        self.vocab = vocab
        self.shuffle = param_config.get_shuffle
        self.model_path = model_save_path
        self.data_inter = Data_Inter(self.batch_size)  # 迭代器。
        if intent2id is not None:
            self.intent2id = intent2id
            self.intent_counts = len(self.intent2id)
            print('self.intent_counts:', self.intent_counts)
            print('self.intent2id:', self.intent2id)
        if label2tag is not None:
            self.label2tag = label2tag
        if id2intent is not None:
            self.id2intent = id2intent

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.loss_op()
        self.trainstep_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.intent_targets = tf.placeholder(tf.int32, [self.batch_size],  # 真正的意图
                                             name='intent_targets')  # 16

    def lookup_layer_op(self):
        """
        将词的one-hot形式表示成词向量的形式，词向量这里采用随机初始化的形式，显然可以使用w2c与训练的词向量。
        """
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)  # 只有当训练的时候droup才会起作用。

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm", reuse=tf.AUTO_REUSE):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=self.word_embeddings,
                        sequence_length=self.sequence_lengths,
                        dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
            encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        with tf.variable_scope("proj", reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],  # 实体的个数
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            w_intent = tf.get_variable(name="W_intent",
                                       shape=[2 * self.hidden_dim, self.intent_counts],  # intent的个数
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       dtype=tf.float32)
            b_intent = tf.get_variable(name="b_intent",
                                       shape=[self.intent_counts],
                                       initializer=tf.zeros_initializer(),
                                       dtype=tf.float32)

            # slot
            s = tf.shape(output)  # shape is (time_step, batch_size, 2*dim)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])  # shape = (time_step*batch_size, 2*dim)
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])  # shape is (time_step, batch_size)

            # intent
            intent_logits = tf.add(tf.matmul(encoder_final_state_h, w_intent), b_intent)  # 得到意图的识别
            self.intent = tf.argmax(intent_logits, axis=1)
            # 定义intent分类的损失
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.intent_targets,
                                                                                      depth=self.intent_counts,
                                                                                      dtype=tf.float32),
                                                                    logits=intent_logits)
            self.loss_intent = tf.reduce_mean(cross_entropy)

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood) + self.loss_intent  # 联合损失

    def trainstep_op(self):
        """
        训练节点.
        """
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)  # 全局训批次的变量，不可训练。
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    @staticmethod
    def pad_sequences(sequences, pad_mark=0, slots=False):
        """
        批量的embedding，其中rowtext embedding的长度要与slots embedding的长度一致，不然使用crf时会出错。
        :param sequences: 批量的文本格式[[], [], ......, []]，其中子项[]里面是一个完整句子的embedding（索引。）
        :param pad_mark:  长度不够时，使用何种方式进行padding
        :param slots:  文本对应的槽。
        :return:
        """
        max_len = max(map(lambda x: len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            if slots:
                # seq_ = seq[:max_len] + [pad_mark] * max(max_len + 1 - len(seq), 0)  # 求得最大的索引长度。
                seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)  # 求得最大的索引长度。
            else:
                seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)  # 求得最大的索引长度。
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    def train(self, log_file=None):
        """
            数据由一个外部迭代器提供。
        """
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_index in range(0, self.epoch_num, 1):
                batches_recording = 0
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                batches_recording += 1
                while batches_recording <= self.data_inter.num_batches:  # 迭代器内部是没有设置结束标志。
                    batches_recording += 1
                    sentence, slots, intents = self.data_inter.next()  # 迭代器，每次取出一个batch块.
                    feed_dict, _ = self.get_feed_dict(sentence, slots, intents, self.lr, self.dropout_keep_prob)
                    _, loss_train, step_num_ = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict)
                    if batches_recording % 2 == 0:
                        if log_file is not None:
                            log_file.write('time:'.__add__(start_time).__add__('\tepoch: ').
                                           __add__(str(epoch_index + 1)).__add__('\tstep:').
                                           __add__(str(batches_recording + epoch_index * self.data_inter.num_batches)).
                                           __add__('\tloss:').__add__(str(loss_train)).__add__('\n'))

                        print('time {} epoch {}, step {}, loss: {:.4}'.
                              format(start_time, epoch_index + 1, batches_recording + epoch_index *
                                     self.data_inter.num_batches, loss_train))

                    check_multi_path(self.model_path)
                    saver.save(sess, self.model_path, global_step=self.data_inter.num_batches * (epoch_index + 1))
            if log_file is not None:
                log_file.close()

    def get_feed_dict(self, seqs, labels=None, intents=None, lr=None, dropout=None):
        """

        :param seqs:  训练的batch块
        :param labels:  实体标签
        :param intents:  意图标签
        :param lr:  学利率
        :param dropout:  活跃的节点数，全连接层
        :return: feed_dict  训练数据
        """
        # seqs的格式： [[], [],......[]]，其中每个列表里面都是一个个索引值，labels也是一样的
        # 这里需要注意的是，atis 数据集中有坑，尤其是在使用crf时会出现，因为在对每一句话进行标注的时候，slot序列是不会对
        # row text的最后一个结束符
        word_ids, seq_len_list = Bi_LSTM_Crf.pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids,  # embedding到同一长度
                     self.sequence_lengths: seq_len_list,  # 实际长度。
                     }
        if intents is not None:
            feed_dict[self.intent_targets] = intents
        if labels is not None:
            labels_, _ = Bi_LSTM_Crf.pad_sequences(labels, pad_mark=0, slots=True)
            feed_dict[self.labels] = labels_  # embedding到当前批的最大长度
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict, seq_len_list

    def task_intent_slots(self, sess, sent):
        """

        :param sess:
        :param sent:
        :return:
        """
        # [[  2  29 137  23 165 242  52   7  14   5 205 206  12]]
        sent_ = np.array([self.data_inter.sentence2index(sent, self.vocab)])  # 句子到index的映射
        label_index, _cur_intent = self.predict(sess, sent_)
        label_index = label_index[0]
        predicted_slots = list(map(lambda y: self.label2tag[y],
                                   list(filter(lambda x: not self.label2tag[x].__eq__('O'), label_index))))
        all = list(map(lambda y: self.label2tag[y],label_index))
        print('all:', all)
        coresponding_words = []
        for index, cur_slot_index in enumerate(label_index):
            if not self.label2tag[cur_slot_index].__eq__('O'):
                coresponding_words.append(sent[index])
        assert len(predicted_slots) == len(coresponding_words), \
            "the slots predicted can't match the value in terms of len "
        # print('coresponding_words:', coresponding_words)
        res = []
        for cur_slot, cur_slot_value in zip(predicted_slots, coresponding_words):
            res.append([cur_slot, cur_slot_value])
        return res, self.id2intent[_cur_intent[0] + 1]

    def predict(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            cur_intent, logits, transition_params = sess.run([self.intent, self.logits, self.transition_params],
                                                             feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, cur_intent
        else:
            print('only labeling not classify')

"""
import time
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from data_pro.data_parse import Data_Inter
from utils import check_multi_path


class Bi_LSTM_Crf(object):
    def __init__(self, param_config, embeddings, tag2label, vocab, model_save_path, intent2id=None):
        self.batch_size = param_config.get_batch_size
        self.epoch_num = param_config.get_epoch
        self.hidden_dim = param_config.get_hidden_dim
        self.embeddings = embeddings
        self.CRF = param_config.get_crf
        self.update_embedding = param_config.get_update_embedding
        self.dropout_keep_prob = param_config.get_dropout
        self.optimizer = param_config.get_optimizer
        self.lr = param_config.get_lr
        self.clip_grad = param_config.get_clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = param_config.get_shuffle
        self.model_path = model_save_path
        self.data_inter = Data_Inter(self.batch_size)  # 迭代器。
        self.intent2id = intent2id
        self.intent_counts = len(self.intent2id)
        # self.result_path = paths['result_path']

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.loss_op()
        self.trainstep_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        self.intent_targets = tf.placeholder(tf.int32, [self.batch_size],  # 真正的意图
                                             name='intent_targets')  # 16

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)  # 只有当训练的时候droup才会起作用。

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=self.word_embeddings,
                        sequence_length=self.sequence_lengths,
                        dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
            encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],  # 实体的个数
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            w_intent = tf.get_variable(name="W_intent",
                                       shape=[2 * self.hidden_dim, self.intent_counts],  # intent的个数
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       dtype=tf.float32)
            b_intent = tf.get_variable(name="b_intent",
                                       shape=[self.intent_counts],
                                       initializer=tf.zeros_initializer(),
                                       dtype=tf.float32)

            # slot
            s = tf.shape(output)  # shape is (time_step, batch_size, 2*dim)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])  # shape = (time_step*batch_size, 2*dim)
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])  # shape is (time_step, batch_size)

            # intent
            intent_logits = tf.add(tf.matmul(encoder_final_state_h, w_intent), b_intent)  # 得到意图的识别
            self.intent = tf.argmax(intent_logits, axis=1)
            # 定义intent分类的损失
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.intent_targets,
                                                                                      depth=self.intent_counts,
                                                                                      dtype=tf.float32),
                                                                    logits=intent_logits)
            self.loss_intent = tf.reduce_mean(cross_entropy)

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood) + self.loss_intent  # 联合损失

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)  # 全局训批次的变量，不可训练。
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    @staticmethod
    def pad_sequences(sequences, pad_mark=0, slots=False):
        max_len = max(map(lambda x: len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            if slots:
                seq_ = seq[:max_len] + [pad_mark] * max(max_len + 1 - len(seq), 0)  # 求得最大的索引长度。
            else:
                seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)  # 求得最大的索引长度。
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    def train(self, log_file=None):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_index in range(0, self.epoch_num, 1):
                batches_recording = 0
                start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                batches_recording += 1
                while batches_recording <= self.data_inter.num_batches:  # 迭代器内部是没有设置结束标志。
                    batches_recording += 1
                    sentence, slots, intents = self.data_inter.next()  # 迭代器，每次取出一个batch块.
                    feed_dict, _ = self.get_feed_dict(sentence, slots, intents, self.lr, self.dropout_keep_prob)
                    _, loss_train, step_num_ = sess.run([self.train_op, self.loss, self.global_step], feed_dict=feed_dict)
                    if batches_recording % 2 == 0:
                        if log_file is not None:
                            log_file.write('time:'.__add__(start_time).__add__('\tepoch: ').
                                           __add__(str(epoch_index + 1)).__add__('\tstep:').
                                           __add__(str(batches_recording + epoch_index * self.data_inter.num_batches)).
                                           __add__('\tloss:').__add__(str(loss_train)))

                        print('time {} epoch {}, step {}, loss: {:.4}'.
                              format(start_time, epoch_index + 1, batches_recording + epoch_index *
                                     self.data_inter.num_batches, loss_train))

                    check_multi_path(self.model_path)
                    saver.save(sess, self.model_path, global_step=self.data_inter.num_batches * (epoch_index + 1))
            if log_file is not None:
                log_file.close()

    def get_feed_dict(self, seqs, labels=None, intents=None, lr=None, dropout=None):
        # seqs的格式： [[], [],......[]]，其中每个列表里面都是一个个索引值，labels也是一样的
        # 这里需要注意的是，atis 数据集中有坑，尤其是在使用crf时会出现，因为在对每一句话进行标注的时候，slot序列是不会对
        # rowtwxt的最后一个结束符
        word_ids, seq_len_list = Bi_LSTM_Crf.pad_sequences(seqs, pad_mark=0)
        feed_dict = {self.word_ids: word_ids,  # embedding到同一长度
                     self.sequence_lengths: seq_len_list,  # 实际长度。
                     self.intent_targets: intents}
        if labels is not None:
            labels_, _ = Bi_LSTM_Crf.pad_sequences(labels, pad_mark=0, slots=True)
            feed_dict[self.labels] = labels_  # embedding到当前批的最大长度
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict, seq_len_list

"""
