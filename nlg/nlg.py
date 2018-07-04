class NLG(object):
    def __init__(self):
        pass

    def generate_sentence(self, action, terminate, task_id):
        if task_id == 1:
            if terminate:
                if action == 101:
                    return '即将为您播放'
            else:
                if action == 0:
                    return '请问您要看哪个频道？'
        return '默认回复'