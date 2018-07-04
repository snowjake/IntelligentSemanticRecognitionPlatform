class Dictionry:
    def __init__(self, task):
        self.replay_slot = ['channelname', 'data', 'time', 'program']
        self.live_slot = ['channelname']
        self.order_slot = ['videoName','category','modifier','person']
        self.cmd_slot=['cmdValue','cmdParam']
        self.dictory = self.inint_dictory(task)

    def inint_dictory(self, task):
        if task == 'replay':
            self.dictory = {j: i for (i, j) in enumerate(self.replay_slot)}
            self.action = {i: j for (i, j) in enumerate(self.replay_slot)}
        if task == 'live':
            self.dictory = {j: i for i, j in enumerate(self.live_slot)}

            self.action = {i: j for i, j in enumerate(self.live_slot)}
        if task == 'order':
            self.dictory = {j: i for i, j in enumerate(self.order_slot)}
            self.action = {i: j for i, j in enumerate(self.order_slot)}
        if task == 'cmd':
            self.dictory =  {j: i for i, j in enumerate(self.cmd_slot)}
        return self.dictory