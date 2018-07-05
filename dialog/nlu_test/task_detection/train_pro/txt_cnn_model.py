import tensorflow as tf


class TxtModel:
    def __init__(self, sentence_length=100, num_class=4, vocabu=1250, signal_word_dim=64):
        self.sentence_length = sentence_length  # 句子的长度。句子长度不超过sentence_length，预处理过程中已padding
        self.num_class = num_class  # 实际分类的类别
        self.vocabu = vocabu  # 字典的长度
        self.signal_word_dim = signal_word_dim  # 单个词的维度表示，这里默认为64，表示使用64维的向量就可以表示一个词。
        self.input = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_length])  # 句子个数 * 句子的嵌入。
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, self.num_class])
        self.dropout = tf.placeholder(dtype=tf.float32)
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=None)

    def create_model(self, scope=None):
        """
        网络的创建
        :param scope: 变量所属的域
        :return:
        """
        if scope is None:
            scope = 'txt_model_scope'
        with tf.variable_scope(name_or_scope=scope, default_name='txt_model_scope'):
            with tf.device('/cpu:0'):
                embedding = tf.get_variable('embedding', [self.vocabu, self.signal_word_dim])  # 也可以直接使用w2c初始化
                embedding_inputs = tf.nn.embedding_lookup(embedding, self.input)
                # embedding_inputs-300*64   filter-256   kernel_size-5
                conv = tf.layers.conv1d(embedding_inputs, 256, 5, name='conv')  # shape=(?, 296, 256)
                # print('conv:', conv)  # shape=(?, 296, 256)
                # 参考了14年的那篇论文 shape=(?, 256)
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')  # 拿到每一个filter对应的向量最大值，然后拼接
                # 全连接层，后面接dropout以及relu激活
                fc = tf.layers.dense(gmp, 128, name='fc1')
                # fc = tf.contrib.layers.dropout(fc, self.dropout)
                # fc = tf.nn.relu(fc, name='relu_fc1')
                self.fc = tf.contrib.layers.dropout(fc, self.dropout)
                self.fc = tf.nn.relu(self.fc, name='relu_fc1')
                self.logits = tf.layers.dense(self.fc, self.num_class, name='fc2')
                self.label_pred = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别
                # 损失函数，交叉熵
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
                self.loss = tf.reduce_mean(cross_entropy)
                self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                # 准确率
                correct_pred = tf.equal(tf.argmax(self.label, 1), self.label_pred)  # [True, False,.......]
                self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # 取平均值。
