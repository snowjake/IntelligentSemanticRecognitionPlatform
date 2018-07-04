from keras.models import Sequential
from keras.layers import Dense
from state_tacker import *
from Dictionry import *
import numpy as np
class DM:
    def __init__(self, task):
        self.task = task
        if self.task == 'replay':
            self.dictory = Dictionry(self.task)
            # self.state_tacker=state_tacker(self.dictory.dictory)
            self.wights = 'mydiaaaaa.h5'
            self.state_tacker = state_tacker(self.dictory.dictory)
            self.naction = len(self.state_tacker.state)
            self.model = self.model(self.wights, self.naction)

            # self.model=self.model(self.wights)
        if self.task == 'order':
            self.dictory = Dictionry(self.task)
            self.wights = 'order.h5'
            self.state_tacker = state_tacker(self.dictory.dictory)
            self.naction = len(self.state_tacker.state)
            self.model = self.model(self.wights, self.naction)
        if self.task == 'live':
            self.dictory = Dictionry(self.task)
            self.state_tacker = state_tacker(self.dictory.dictory)
            self.naction = len(self.state_tacker.state)
            self.model = self.model(self.wights, self.naction)
            self.wights = 'order.h5'
        if self.task=='cmd':
            self.dictory = Dictionry(self.task)
            self.state_tacker = state_tacker(self.dictory.dictory)


    def call_api(self, task):
        print(self.task)

    def which_intention(self, intention, slot):
        if intention == 'inform':
            state, values = self.state_tacker.updata_state(slot)
            return state, values
        if intention=='cmd':
            state, values = self.state_tacker.updata_state(slot)
            return state, values

    def model(self, weight, naction):
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(naction, activation='linear'))
        model.load_weights(weight)
        return model
    '''def cmd_control(self,slot):
        state, values = self.state_tacker.updata_state(slot)
        return state, values'''
    def select_action(self, state):
        if self.judge_end(state):
            action = self.model.predict(np.array([state])).argmax()
            return (self.dictory.action[action])
        else:
            return 'done'

    def judge_end(self, state):
        flag = 1
        if self.task == 'replay':
            if (state[0] == 1 and state[1] == 1 and state[2] == 1):
                flag = 0
        if self.task == 'order':
            if (state[0] == 1):
                flag = 0
        if self.task == 'live':
            if (state[0] == 1):
                flag = 0
        return flag