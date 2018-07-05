class SpecialParse:
    def __init__(self):
        self.original_data_path = 'tvchannel.txt'
        self.task_data_path = 'task_train.txt'
        self.split_signal = ','
        self.lazy_task_mapping = {
                '点播': 'dianbo',
                '模糊': 'mohu',
                '直播': 'zhibo',
                '回看': 'huikan'
        }

    @property
    def get_original_data(self):
        return self.original_data_path

    @property
    def get_task_data(self):
        return self.task_data_path

    @property
    def get_mapping(self):
        return self.lazy_task_mapping

    def task_data(self):
        # BOS what is sa EOS	O O O B-days_code atis_abbreviation
        # 谁大一点,O O O O O,模糊,其他
        data = open(self.get_original_data, mode='r', encoding='utf-8', errors='ignore').readlines()
        save = open(self.get_task_data, mode='w', encoding='utf-8', newline='')
        # data = list(map(lambda line: [line.split(self.split_signal)[0], line.split(self.split_signal)[2]], data))
        for index, line in enumerate(data):
            cur_data = line.strip().split(self.split_signal)[:]
            print((index, cur_data))
            assert len(cur_data) == 4
            save.write(self.get_mapping[cur_data[2]] + '\t' + cur_data[0] + '\n')
        save.close()

    def task_data_new(self):
        # BOS what is sa EOS	O O O B-days_code atis_abbreviation
        # 谁大一点,O O O O O,模糊,其他
        pre_data = open('task_train.txt', mode='r', encoding='utf-8', errors='ignore').readlines()  # 已经保存的训练数据
        new_data = open('more_bigger_mapping.txt', mode='r', encoding='utf-8', errors='ignore')
        save = open('new_train.txt', mode='w', encoding='utf-8', newline='')
        # data = list(map(lambda line: [line.split(self.split_signal)[0], line.split(self.split_signal)[2]], data))
        for index, (cur_pre, cur_new) in enumerate(zip(pre_data, new_data)):
            cur_pre = cur_pre.strip().split('\t')
            cur_new = (cur_new.strip().replace('[', '').replace(']', '').replace("'", '')).split(',')
            print('cur_new:', cur_new)
            assert len(cur_new) % 2 == 0
            row_text = list(map(lambda x: cur_new[x], range(0, len(cur_new), 2)))
            row2compare = ''.join(row_text).replace(' ', '')
            print((row2compare, cur_pre[1]))
            assert len(row2compare) == len(cur_pre[1]), '不匹配'
            save.write('\t'.join([cur_pre[0], ' '.join(row_text), '\n']))
        save.close()

    def task_data_full(self):
        # BOS what is sa EOS	O O O B-days_code atis_abbreviation
        # 谁大一点,O O O O O,模糊,其他
        pre_data = open('tvchannel.txt', mode='r', encoding='utf-8', errors='ignore').readlines()  # 已经保存的训练数据
        new_data = open('more_bigger_mapping.txt', mode='r', encoding='utf-8', errors='ignore')
        save = open('new_train_full.txt', mode='w', encoding='utf-8', newline='')
        for index, (cur_pre, cur_new) in enumerate(zip(pre_data, new_data)):
            cur_pre = cur_pre.strip().split(',')
            print('cur_pre:', cur_pre)
            _cur_new = (cur_new.strip().replace('[', '').replace(']', '').replace("'", '')).\
                replace(",", '').replace(" ", '')
            print('cur_new:', _cur_new)
            save.write('\t'.join([cur_new.strip(), cur_pre[2].strip(), cur_pre[3].strip(), '\n']))
        save.close()



if __name__ == '__main__':
    so = SpecialParse()
    so.task_data_full()

    ddd = '123456789'
    ff = ddd.replace('123', '')
    print(ff)