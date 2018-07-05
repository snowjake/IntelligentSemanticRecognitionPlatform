class NLG:
    def __init__(self):
        pass

    def to_speaker(self,task,slot):
        if task == 'huikan':
            action = {'time': '您想看几点的呢？', 'startDate': '您想看那一天的呢？', 'channelName': '您想看哪个频道',
                           'name': '您想看哪个节目?', 'done': '马上为您播放'}
        if task == 'order':
            action = {'videoName': '您想看什么节目呢？', 'category': '您想看哪个类型呢？', 'modifier': '您想看什么种类呢',
                           'area': '您想看谁演的呢?', 'done': '马上为您播放'}
        if task=='cmd':
            action={'done':'马上为您执行','NEXTPAGE':'正在切换'}
        if task=='live':
            action={'done':'马上为您切换'}
        print(action[slot['action']])
        return(action[slot['action']])