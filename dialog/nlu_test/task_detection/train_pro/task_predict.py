from __future__ import print_function
import socket
import json
import pickle
import numpy as np
import tensorflow as tf
from nlu_test.task_detection.train_pro.txt_cnn_model import TxtModel

receive_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_s.bind(('127.0.0.2', 50007))


class NetConfig:

    @property
    def __author__(self):
        from functools import reduce
        return reduce(lambda x, y: x + y, self.author_warming)

    # @property
    # def fea_path(self):
    #     return self.fea

    def __init__(self):
        self.author_warming = open('author_info.log', mode='r', encoding='utf-8', errors='ignore').readlines()
        self.cat2id_id2cat_path = '../data/cat2id_id2cat/cat2id_id2cat.pkl'
        self.word2id_in2word_path = '../data/word2id_in2word/word2id_in2word.pkl'

        self.model_path = '../checkpoints/params.model-551'
        # self.fea = '../train_pro/feature_map/feature.fea'  # 特征文件
        self.sentence_length = 100  # maximum length of a signal sentence
        self.word_dim = 64  # the dim of each word, which defined in the network file.
        self.stop = ['”', '“', '、', '。', '，', '──', '……', '（', '）', '？', '《', '》', '<', '>',
                     '！', '......', '.', ',', '；', ';', '%', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                     'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C',
                     'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W', 'X', 'Y', 'Z']

    def get_cat(self):
        try:
            cat_config = pickle.load(open(self.cat2id_id2cat_path, mode='rb'))
            self._cat2index, self._index2cat = cat_config['cat2index'], cat_config['index2cat']
        except Exception:
            raise Exception

    def get_vocab(self):
        try:
            word_config = pickle.load(open(self.word2id_in2word_path, mode='rb'))
            self._word2index, self._index2word = word_config['word2index'], word_config['index2word']
        except Exception:
            raise Exception

    def lunch_model(self):
        self.get_cat()
        self.get_vocab()  # can be optimised
        # self.model = TxtModel(vocabu=len(self._index2word))
        self.model = TxtModel(vocabu=389)
        self.model.create_model()  # load the net params
        self.session = tf.Session()
        # self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=self.model_path)  # awake the network

    def predict(self, txt):
        # padding
        # 黑放累!!!!!!------>['黑', '放', '累']------>[6, 99, 666]
        txt = txt.strip().split()
        pur_index = list(map(lambda x: self._word2index.get(x, 0), list(filter(lambda x: x not in self.stop, txt))))

        if len(pur_index) < self.sentence_length:
                    # padding extend mat change the original sequences
                    pur_index.extend([0] * (self.sentence_length - len(pur_index)))
        else:
            pur_index = pur_index[: self.sentence_length]  # kill the tail。  may trouble
        feed_dict = {
            self.model.input: np.array(pur_index)[np.newaxis, :],
            self.model.dropout: 0.66
        }
        # return index2cat[self.session.run(self.model.label_pred, feed_dict=feed_dict)[0]]
        # return self.session.run(self.model.fc, feed_dict=feed_dict)  # 得到特征，然后计算分数。
        # return self.session.run(self.model.fc, feed_dict=feed_dict)  # 得到特征，然后计算分数。
        return self.session.run(self.model.label_pred, feed_dict=feed_dict)[0]  # 得到预测类别。

    @staticmethod
    def consin_score(saved_model, test_feature):
        numclass = len(saved_model)
        score_vectore = []
        score_label = []
        for index in range(numclass):
            label, enrollment_model = saved_model[index][:]
            enrollment_model = enrollment_model[np.newaxis, :]
            n_feature = np.sqrt(test_feature)
            m_feature = np.sqrt(enrollment_model)
            m_norm = m_feature / np.sqrt(np.sum(np.square(m_feature), axis=1, keepdims=True))
            n_norm = n_feature / np.sqrt(np.sum(np.square(n_feature), axis=1, keepdims=True))
            similar_score = np.dot(n_norm, m_norm.T)
            score_vectore.append(similar_score)
            score_label.append(label)
        return score_vectore, score_label


if __name__ == '__main__':
    cnn_model = NetConfig()
    cnn_model.lunch_model()  # load dict
    print(cnn_model.__author__)
    while True:
        data, address = receive_s.recvfrom(1024)
        str_to_dict = json.loads(data.decode())
        print('row_text:', str_to_dict['row_text'])
        new_row_text = ' '.join(str_to_dict['row_text'])
        print('new_row_text:', new_row_text)
        # task = cnn_model._index2cat[cnn_model.predict(str_to_dict['row_text'][0])]
        task = cnn_model._index2cat[cnn_model.predict(new_row_text)]
        print("task:", task)
        receive_s.sendto(json.dumps({'task_res': task},
                                    ensure_ascii=False).encode(), ('127.0.0.2', 50008))

    # # 仅限于task_detection测试
    # ssss = input()
    # print('signal task:', cnn_model._index2cat[cnn_model.predict(ssss)])
