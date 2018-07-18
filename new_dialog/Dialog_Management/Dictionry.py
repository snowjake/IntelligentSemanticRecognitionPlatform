class Dictionry:
    def __init__(self, task):
        self.replay_slot = ['channelName', 'startDate', 'startTime', 'name','endTime']
        self.live_slot = ['channelname']
        self.order_slot = ['videoName','category','modifier','area','persons',
                           'season','episode','startyear','endyear']
        self.cmd_slot=['cmdValue','cmdParam']
        self.dictory = self.inint_dictory(task)

    def inint_dictory(self, task):
        if task == 'huikan':
            self.dictory = {j: i for (i, j) in enumerate(self.replay_slot)}
            #self.action = {i: j for (i, j) in enumerate(self.replay_slot)}
        if task == 'live':
            self.dictory = {j: i for i, j in enumerate(self.live_slot)}

            #self.action = {i: j for i, j in enumerate(self.live_slot)}
        if task == 'dianbo':
            self.dictory = {j: i for i, j in enumerate(self.order_slot)}
            #self.action = {i: j for i, j in enumerate(self.order_slot)}
        if task == 'cmd':
            self.dictory =  {j: i for i, j in enumerate(self.cmd_slot)}
        return self.dictory