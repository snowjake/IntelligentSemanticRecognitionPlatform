import pickle
import os
import sys
from random import sample
import numpy as np

class DataParse:
    def __init__(self):
        # self.train_data_path = '../../dataset/train.txt'
        self.train_data_path = '../../dataset/new_train_intent_slot_labeling.txt'  # 新的数据
        self.slot2id_id2sot = '../data_parse/slot2id_id2sot'
        self.intent2id_id2intent = '../data_parse/intent2id_id2intent'
        self.sentence_slot_intent_path = '../data_parse/sentence_slot_intent'
        self.vocb_path = '../data_parse/vocb'

    def read_corpus(self, corpus_path=None, save=True):
        """
            获取语料，路径
        """
        if corpus_path is None:
            assert 'path empty'
        try:
            with open(corpus_path, mode='r', encoding='utf-8') as fr:
                lines = fr.readlines()
            sentence_slot_intent = []
            slot2id_id2slot = {}
            intent2id_id2intent = {}
            slot2id = {}
            id2slot = {}
            intent2id = {}
            id2intent = {}
            for line in lines:
                # [char, label] = line.strip().split()
                sentence, slot_intent = line.strip().split('\t')
                slots_intent = slot_intent.split()
                slots, intent = slots_intent[:-1], slots_intent[-1]
                print('fffffff:', (sentence, slots, intent))
                sentence_slot_intent.append((sentence.split(), slots, [intent]))  # 句子--槽--意图
                # sentence_slot_intent.append((list(sentence), slots, [intent]))  # 句子--槽--意图
                for cur_slot in slots:
                    if slot2id.get(cur_slot, 10000) == 10000:
                        slot2id[cur_slot] = len(slot2id)
                        id2slot[len(slot2id) - 1] = cur_slot
                if intent2id.get(intent, 10000) == 10000:
                    intent2id[intent] = len(intent2id)
                    id2intent[len(intent2id)] = intent
            slot2id_id2slot['slot2id'] = slot2id
            slot2id_id2slot['id2slot'] = id2slot
            intent2id_id2intent['intent2id'] = intent2id
            intent2id_id2intent['id2intent'] = id2intent
            print(slot2id_id2slot)
            if save:
                pickle.dump(sentence_slot_intent, open(self.sentence_slot_intent_path, mode='wb'))
                pickle.dump(slot2id_id2slot, open(self.slot2id_id2sot, mode='wb'))
                pickle.dump(intent2id_id2intent, open(self.intent2id_id2intent, mode='wb'))
            return sentence_slot_intent, slot2id_id2slot, intent2id_id2intent
        except Exception:
            raise Exception

    def old_creat_vocab(self, data=None, save=True):
        if not os.path.exists(self.vocb_path):
            vocab = {}
            vocab['<UNK>'] = 1  # 未知的词汇
            vocab['<PAD>'] = 0  # 需要被填充的标记
            if data is not None:
                assert isinstance(data, list) and isinstance(data[0], tuple)
                for sentence, _, _ in data:
                    sentence = sentence.split()
                    print('sentence', sentence)
                    for cut_word in sentence:
                        if vocab.get(cut_word, 0) == 0:
                            vocab[cut_word] = len(vocab)
                if save:
                    pickle.dump(vocab, open(self.vocb_path, mode='wb'))
                return vocab
            else:
                print('data empty......')
        else:
            sys.stdout.write('vocab exists......')
            return pickle.load(open(self.vocb_path, mode='rb'))

    def creat_vocab(self, data=None, save=True):
        if not os.path.exists(self.vocb_path):
            vocab = {}
            vocab['<UNK>'] = 1  # 未知的词汇
            vocab['<PAD>'] = 0  # 需要被填充的标记
            if data is not None:
                assert isinstance(data, list) and isinstance(data[0], str)
                for sentence_slot_intent in data:
                    all_splited = sentence_slot_intent.strip().split('\t')
                    print(all_splited)
                    assert len(all_splited) == 2
                    print('all_splited[0]:', all_splited[0])
                    data_ss = all_splited[0].split()
                    print('data_ss:', data_ss)
                    for cut_word in data_ss:
                        if vocab.get(cut_word, -1) == -1:
                            vocab[cut_word] = len(vocab)
                if save:
                    pickle.dump(vocab, open(self.vocb_path, mode='wb'))
                return vocab
            else:
                print('data empty......')
        else:
            sys.stdout.write('vocab exists......')
            return pickle.load(open(self.vocb_path, mode='rb'))

    def random_embedding(self, embedding_dim, word_num):
        """
        随机的生成word的embedding，这里如果有语料充足的话，可以直接使用word2vec蓄念出词向量，这样词之间的区别可能更大。
        :param embedding_dim:  词向量的维度。
        :return: numpy format array. shape is : (vocab, embedding_dim)
        """
        # if vocb_paths is None:
        #     vocab_creatation = pickle.load(open(self.vocb_path, mode='rb'))
        # else:
        #     vocab_creatation = pickle.load(open(vocb_paths, mode='rb'))
        embedding_mat = np.random.uniform(-0.25, 0.25, (word_num, embedding_dim))
        embedding_mat = np.float32(embedding_mat)
        return embedding_mat
class Data_Inter:
    """
    生成训练数据
    """
    def __init__(self, batch_size):
        self.sentence_slot_intent_path = r'C:\Users\Administrator\Desktop\dx_joint_entity_recognition_intent_detection\atis_entity_recognition\data_parse\sentence_slot_intent'
        self.slot2id_id2sot = r'C:\Users\Administrator\Desktop\dx_joint_entity_recognition_intent_detection\atis_entity_recognition\data_parse\slot2id_id2sot'
        self.intent2id_id2intent = r'C:\Users\Administrator\Desktop\dx_joint_entity_recognition_intent_detection\atis_entity_recognition\data_parse\intent2id_id2intent'
        self.vocb_path = r'C:\Users\Administrator\Desktop\dx_joint_entity_recognition_intent_detection\atis_entity_recognition\data_parse\vocb'
        self.batch_size = batch_size
        self.index = 0
        # self.initializer()
        if os.path.exists(self.sentence_slot_intent_path):
            # 格式[['Hei], 'fun', 'lei'], ['O', 'B', 'O'], ['Love']]------> rowtext, slots, intent
            self.sentence_slot_intent = np.array(pickle.load(open(self.sentence_slot_intent_path, mode='rb')))
            self.end = len(self.sentence_slot_intent)
            print('self.end:', self.end)
            self.num_batches = self.end // self.batch_size
            self.shuffle = sample(range(0, self.end, 1), self.end)
        else:
            print('train data is empty......')

        if os.path.exists(self.vocb_path):
            self.vocab = pickle.load(open(self.vocb_path, mode='rb'))
        else:
            print('vocab is empty......')

        if os.path.exists(self.slot2id_id2sot):
            self.intent2index = pickle.load(open(self.slot2id_id2sot, mode='rb'))['slot2id']
        else:
            print('slots mapping must be provided......')

        if os.path.exists(self.intent2id_id2intent):
            self.intent2id = pickle.load(open(self.intent2id_id2intent, mode='rb'))['intent2id']
        else:
            print('intent mapping must be provided......')

    def next(self):
        sentence = []
        slots = []
        intents = []
        if self.index + self.batch_size <= self.end:
            it_data = self.sentence_slot_intent[self.shuffle[self.index: self.index + self.batch_size], :]  # 迭代数据
            self.index = self.end - self.index - self.batch_size
        if self.index + self.batch_size == self.end:
            self.shuffle = sample(range(0, self.end, 1), self.end)
        if self.index + self.batch_size > self.end:
            it_data = self.sentence_slot_intent[self.shuffle[self.index: self.end], :]  # 随机选取
            self.index = 0
            remain = self.sentence_slot_intent[self.shuffle[self.index: self.index + self.batch_size], :]  # 剩余
            it_data = np.concatenate((it_data, remain), axis=0)
        for cur_sentences, cur_slots, cur_intent in it_data:
            sentence.append(self.sentence2index(cur_sentences, self.vocab))
            slots.append(self.slots2index(cur_slots, self.intent2index))
            intents.append(self.intent2id[cur_intent[-1]])
        return np.array(sentence), np.array(slots), np.array(intents)

    def sentence2index(self, sen, vocab):
        # print('sen sne:', sen)
        # sen = sen.split()
        assert isinstance(sen, list) and len(sen) > 0
        assert isinstance(vocab, dict) and len(vocab) > 0
        sen2id = []
        for cur_sen in sen:
            sen2id.append(vocab.get(cur_sen, 0))  # 如果找不到，就用0代替。
        return sen2id

    def slots2index(self, cur_slots, mapping):
        assert isinstance(cur_slots, list) and len(cur_slots) > 0 and hasattr(cur_slots, '__len__')
        assert isinstance(mapping, dict) and len(mapping) > 0
        cur_slot2index_mapping = []
        for cur_slot in cur_slots:
            cur_slot2index_mapping.append(mapping[cur_slot])
        return cur_slot2index_mapping

    #
    # def intents2ind(self, cur_intent, intentmapping):
    #     assert isinstance(cur_intent, list) and len(cur_intent) > 0 and hasattr(cur_intent, '__len__')
    #     assert isinstance(intentmapping, dict) and len(intentmapping) > 0
    #     signal_intent_mapping = []