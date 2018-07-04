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
        if task_id == 2:
            if terminate:
                if action == 101:
                    return '即将为您播放'
            else:
                if action == 0:
                    return '请问您要看哪个频道？'
                elif action == 1:
                    return '请问您要看哪一天的呢？'
                elif action == 2:
                    return '请问您要看几点的节目呢'
                elif action == 3:
                    return '请问您要看哪个节目呢'
        return '默认回复'