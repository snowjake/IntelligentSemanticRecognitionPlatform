import os
import json
import pickle
import socket
import numpy as np
import tensorflow as tf
from nlu_test.atis_entity_recognition.filter_parten import FilterRule
from nlu_test.slot_value_check.slot_value_check import SlotValueCheck, SlotLazyMapping
from nlu_test.atis_entity_recognition.bi_lstm_model import Bi_LSTM_Crf
from nlu_test.atis_entity_recognition.data_pro.data_parse import DataParse


class Config:
    def __init__(self):
        self.root_data_path = 'nlu_test/atis_entity_recognition/data_path'
        self.batch_size = 32
        self.epoch = 100
        self.hidden_dim = 100
        self.optimizer = 'Adam'
        self.CRF = True  # 在BI-LSTM的输出层使用CRF进行标注
        self.lr = 0.001
        self.clip = 5.0  # 防止梯度爆炸
        self.dropout = 0.9
        self.update_embedding = True  # 训练的时候更新映射
        self.pretrain_embedding = False  # 词向量的初始化方式，随机初始化
        self.embedding_dim = 100  # 词向量的维数
        self.shuffle = True  # 打乱训练数据
        self.log = 'train_log.txt'

    @property
    def get_log(self):
        return self.log

    @property
    def get_root_data_path(self):
        return self.root_data_path

    @property
    def get_batch_size(self):
        return self.batch_size

    @property
    def get_epoch(self):
        return self.epoch

    @property
    def get_hidden_dim(self):
        return self.hidden_dim

    @property
    def get_optimizer(self):
        return self.optimizer

    @property
    def get_crf(self):
        return self.CRF

    @property
    def get_lr(self):
        return self.lr

    @property
    def get_clip(self):
        return self.clip

    @property
    def get_dropout(self):
        return self.dropout

    @property
    def get_update_embedding(self):
        return self.update_embedding

    @property
    def get_pretrain_embedding(self):
        return self.pretrain_embedding

    @property
    def get_embedding_dim(self):
        return self.embedding_dim

    @property
    def get_shuffle(self):
        return self.shuffle


host = '127.0.0.2'
receive_port = 50008
receive_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive_s.bind((host, receive_port))


# class Train:
class NLU(object):
    def __init__(self):
        self.config = Config()
        self.dataparse = DataParse()  # 数据预处理相关。
        self.word2id = pickle.load(open('nlu_test/atis_entity_recognition/data_parse/vocb', mode='rb'))  # 获取本地存放的字典。
        self.tang2index = pickle.load(open('nlu_test/atis_entity_recognition/data_parse/slot2id_id2sot', mode='rb'))['slot2id']
        self.index2tang = pickle.load(open('nlu_test/atis_entity_recognition/data_parse/slot2id_id2sot', mode='rb'))['id2slot']
        self.intent2id = pickle.load(open('nlu_test/atis_entity_recognition/data_parse/intent2id_id2intent', mode='rb'))['intent2id']
        self.id2intent = pickle.load(open('nlu_test/atis_entity_recognition/data_parse/intent2id_id2intent', mode='rb'))['id2intent']
        self.filter_rule = FilterRule()  # 过滤规则
        if not self.config.get_pretrain_embedding:
            self.embeddings = self.dataparse.random_embedding(self.config.get_embedding_dim, len(self.word2id))
        else:
            pre_trained_word_model_path = os.path.join('data_path', 'pre_trained_word.pkl')
            assert os.path.exists(pre_trained_word_model_path), '暂时没有预训练好的词向量'
            self.embeddings = np.array(np.load(pre_trained_word_model_path), dtype='float32')  # 否则加载预训练好词向量。
        self.model_same_path = self.config.get_root_data_path.__add__('model_save/checkpoints/')
        self.huikan_slot_check = SlotValueCheck()  # 对回看槽的合法性的判断。并不做逻辑上的反馈。
        self.slot_mapping = SlotLazyMapping()  # 槽的映射 任务/槽
        self.init()

    def init(self):
        self.model = Bi_LSTM_Crf(self.config,
                            self.embeddings,
                            self.tang2index,
                            self.word2id,
                            self.model_same_path,
                            self.intent2id,
                            self.index2tang,
                            self.id2intent
                            )
        self.model.build_graph()

        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        # 加载 intent detection 和 slot recognition模型参数
        ckpt_file = tf.train.latest_checkpoint(self.config.get_root_data_path.__add__('model_save/checkpoints/'))
        saver.restore(self.sess, ckpt_file)
        # with tf.Session() as sess:
        #     saver = tf.train.Saver()
        #     # 加载 intent detection 和 slot recognition模型参数
        #     ckpt_file = tf.train.latest_checkpoint(self.config.get_root_data_path.__add__('model_save/checkpoints/'))
        #     saver.restore(sess, ckpt_file)


    def predict(self):
        model = Bi_LSTM_Crf(self.config,
                            self.embeddings,
                            self.tang2index,
                            self.word2id,
                            self.model_same_path,
                            self.intent2id,
                            self.index2tang,
                            self.id2intent
                            )
        model.build_graph()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            # 加载 intent detection 和 slot recognition模型参数
            ckpt_file = tf.train.latest_checkpoint(self.config.get_root_data_path.__add__('model_save/checkpoints/'))
            saver.restore(sess, ckpt_file)

            while True:
                row_text = input()
                if row_text == '' or row_text.isspace():
                    break
                else:
                    # task detect result
                    row_text = self.filter_rule.cut_sentence(row_text)
                    # receive_s.sendto(json.dumps({'row_text': row_text.strip().split()}).encode(), (host, 50007))
                    receive_s.sendto(json.dumps({'row_text': row_text}).encode(), (host, 50007))
                    receve_data, addr = receive_s.recvfrom(1024)
                    receve_data = json.loads(receve_data.decode())
                    # intent detection & slot recognition results
                    tag = [self.model.task_intent_slots(self.sess, row_text), receve_data["task_res"]]
                    slots_intent, task = tag[0], tag[1]
                    slots, intent = slots_intent[0], slots_intent[1]
                    print('task intent slots:', (task, intent, slots))
                    # 检测被填满的槽。
                    marged_slot = {}
                    for ss in slots:
                        slot, value = ss[:]
                        if marged_slot.get(slot.split('_')[1], -1) == -1:
                            marged_slot[slot.split('_')[1]] = value
                        else:
                            marged_slot[slot.split('_')[1]] = marged_slot[slot.split('_')[1]].__add__(value)
                    print('pre:', marged_slot)
                    # 对槽值的格式进行转换，和纠正
                    bad_slots = []
                    for cur_slot, cur_concat_value in zip(marged_slot.keys(), marged_slot.values()):
                        """
                            直播只有两个槽: channelName 和 channelNo。
                            输入 明天。模型的识别结果：task: zhibo slot={'startDate': '明天'},显然标错了，所以就丢掉。
                        """
                        if cur_slot not in self.slot_mapping.map_slot_params_per[self.slot_mapping.pre_mapping[task]]:
                            bad_slots.append(cur_slot)
                        else:
                            if self.slot_mapping.map_config[self.slot_mapping.pre_mapping[task]][cur_slot] is not None:
                                print(cur_slot, cur_concat_value)
                                resss = self.slot_mapping.map_config[self.slot_mapping.pre_mapping[task]][cur_slot]([cur_concat_value])
                                marged_slot[cur_slot] = resss
                    for d in bad_slots:
                        marged_slot.pop(d)
                    return marged_slot

    def rwa_to_slots(self, raw_text):
        # with tf.Session() as sess:
            # task detect result
            row_text = self.filter_rule.cut_sentence(raw_text)
            # receive_s.sendto(json.dumps({'row_text': row_text.strip().split()}).encode(), (host, 50007))
            receive_s.sendto(json.dumps({'row_text': row_text}).encode(), (host, 50007))
            receve_data, addr = receive_s.recvfrom(1024)
            receve_data = json.loads(receve_data.decode())
            # intent detection & slot recognition results
            tag = [self.model.task_intent_slots(self.sess, row_text), receve_data["task_res"]]
            slots_intent, task = tag[0], tag[1]
            slots, intent = slots_intent[0], slots_intent[1]
            print('task intent slots:', (task, intent, slots))
            # 检测被填满的槽。
            marged_slot = {}
            for ss in slots:
                slot, value = ss[:]
                if marged_slot.get(slot.split('_')[1], -1) == -1:
                    marged_slot[slot.split('_')[1]] = value
                else:
                    marged_slot[slot.split('_')[1]] = marged_slot[slot.split('_')[1]].__add__(value)
            print('pre:', marged_slot)
            # 对槽值的格式进行转换，和纠正
            bad_slots = []
            for cur_slot, cur_concat_value in zip(marged_slot.keys(), marged_slot.values()):
                """
                    直播只有两个槽: channelName 和 channelNo。
                    输入 明天。模型的识别结果：task: zhibo slot={'startDate': '明天'},显然标错了，所以就丢掉。
                """
                if cur_slot not in self.slot_mapping.map_slot_params_per[self.slot_mapping.pre_mapping[task]]:
                    bad_slots.append(cur_slot)
                else:
                    if self.slot_mapping.map_config[self.slot_mapping.pre_mapping[task]][cur_slot] is not None:
                        print(cur_slot, cur_concat_value)
                        resss = self.slot_mapping.map_config[self.slot_mapping.pre_mapping[task]][cur_slot]([cur_concat_value])
                        marged_slot[cur_slot] = resss
            for d in bad_slots:
                marged_slot.pop(d)
            res = {}
            res.update({'task': task})
            # res.update({'slots': marged_slot.keys()})
            # res.update({'slots_values': marged_slot.values()})
            res.update({'slot': marged_slot})
            res.update({'intention': intent})
            return res


