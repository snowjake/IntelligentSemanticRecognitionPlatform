from DM import *
class Dialog:
    def __init__(self, task):
        self.task = ' '
        self.history = {}
        self.DM = DM(task)