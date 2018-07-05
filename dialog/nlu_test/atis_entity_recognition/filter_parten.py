import re
from nlu_test.slot_value_check.slot_value_check import SlotValueCheck

class FilterRule:
    def __init__(self):
        self.rule = {
            'rule1': r'[a-zA-Z]+',  # 匹配所由得英文串 如，我想看CCTV一台昨天的焦点访谈----->['CCTV']
            # 匹配所由得数字和时间如，'我想看2018年晚上6点半和6:45分的新闻联播'------> ['2018', '6', '6:45']
            # 'rule2': r'[0-9]+[:]?[0-9]*',  # 6:30的新闻联播2018年的  6:30 2018
            'rule2': r'[0-9]+[:]?[0-9]+',  # 6:30的新闻联播2018年的  6:30
            # 'rule3': r'\d+',  # 6:30的新闻联播2018年的  2018
        }

    def cut_sentence(self, sentence):
        """
         通过正则取切分句子，粒度更大
        :param sentence: 待切分的句子
        :return: 切分后的句子
        """

        # 在句子进入到NLU之前就需要进行转换
        sentence = SlotValueCheck.start_channelname_mapping(sentence)
        sentence = SlotValueCheck.year_time_checking(sentence)

        elements = []
        for cur_rule in self.rule.values():
            elements.extend(re.findall(cur_rule, sentence))

        # 寻找切分出来的子串在原串中的位置
        margin = []
        if len(elements) > 0:
            indexs = {}
            for ind, cur_elem in enumerate(elements):
                indexs[(re.search(cur_elem, sentence).span())] = ind
            indexs = sorted(zip(indexs.keys(), indexs.values()), reverse=False)  # 我想看2018年晚上6点半和6:45分的cctv新闻联播
            char_index = 0
            index_sle = 0
            while char_index < len(sentence):
                while index_sle < len(indexs):
                    if char_index < indexs[index_sle][0][0]:
                        margin.append(sentence[char_index])
                        char_index += 1
                    else:
                        margin.append(elements[indexs[index_sle][1]])
                        char_index = indexs[index_sle][0][1]
                        index_sle += 1
                        break
                if index_sle == len(indexs):
                    break
            latested = indexs[-1][0][1]
            while latested < len(sentence):
                margin.append(sentence[latested])
                latested += 1
        else:
            latested = 0
            while latested < len(sentence):
                margin.append(sentence[latested])
                latested += 1
        return margin

