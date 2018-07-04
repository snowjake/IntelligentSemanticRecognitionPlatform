class Checker(object):

    def __init__(self):
        pass

    def check_channel_name(self, channel_name):
        '''
        检查数据库判断，channel_name是否存在
        :param channel_name:
        :return: 0：channel_name合法，1：channel_name不存在 ...
        '''
        return 0

    def check_channel_id(self, check_channel_id):
        '''
        检查频道号
        1是不是数字
        2范围是否合法
        :param check_channel_id:
        :return: 0：channel_id合法，1：channel_id不是数字，2：channel_id超出范围
        '''