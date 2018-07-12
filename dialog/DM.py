from keras.models import Sequential
from keras.layers import Dense
from state_tacker import *
from Dictionry import *
import numpy as np
import pandas as pd
class DM:
    def __init__(self, task):
        self.task = task
        if self.task == 'huikan':
            self.dictory = Dictionry(self.task)
            # self.state_tacker=state_tacker(self.dictory.dictory)
            #self.wights = 'replay.h5'
            self.wights='replay.csv'
            self.state_tacker = state_tacker(self.dictory.dictory)
            self.naction = len(self.state_tacker.state)
            self.model = self.model(self.wights, self.naction)

            # self.model=self.model(self.wights)
        if self.task == 'dianbo':
            self.dictory = Dictionry(self.task)
            #self.wights = 'order.h5'
            self.wights='order.csv'
            self.state_tacker = state_tacker(self.dictory.dictory)
            self.naction = len(self.state_tacker.state)
            self.model = self.model(self.wights, self.naction)
        if self.task == 'live':
            self.dictory = Dictionry(self.task)
            self.wights = 'live.h5'
            self.state_tacker = state_tacker(self.dictory.dictory)
            self.naction = len(self.state_tacker.state)
            self.model = self.model(self.wights, self.naction)

        if self.task=='cmd':
            self.dictory = Dictionry(self.task)
            self.state_tacker = state_tacker(self.dictory.dictory)


    def call_api(self, task):
        print(self.task)

    def which_intention(self, intention, slot):
        if intention == '告知':
            state, values = self.state_tacker.updata_state(slot)
            return state, values
        if intention=='cmd':
            state, values = self.state_tacker.updata_state(slot)
            return state, values

    def model(self, weight, naction):
        '''model = Sequential()
        model.add(Dense(24, input_dim=naction, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(naction, activation='linear'))
        model.load_weights(weight)'''
        model=pd.read_csv(weight,index_col='Unnamed: 0')
        return model
    '''def cmd_control(self,slot):
        state, values = self.state_tacker.updata_state(slot)
        return state, values'''
    def select_action(self, state):
        if self.judge_end(state):
            #action = self.model.predict(np.array([state])).argmax()
            action = self.model.loc[str(state)].idxmax()
            return (self.dictory.action[int(action)])
        else:
            return 'done'

    def judge_end(self, state):
        flag = 1
        if self.task == 'huikan':
            if (state[0] == 1 and state[1] == 1 and state[2] == 1 and state[3]==1):
                flag = 0
        if self.task == 'dianbo':
            if (state[0] == 1 and state[1] == 1 and state[2] == 1 and state[3]==1):
                flag = 0
        if self.task == 'live':
            if (state[0] == 1):
                flag = 0
        return flag