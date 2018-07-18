from Dialog_Management.state_tacker import *
class action_select:
    def __init__(self,task,intention,state_tacker,slot):
        self.task=task
        self.intention=intention
        self.state_tacker=state_tacker
        self.slot=slot
        self.action=None
        self.state=None
        self.values=None

    def choice_action(self):
        if self.task=='huikan':
            action,state,values=self.select_replay_action
        if self.task=='dianbo':
            action, state, values = self.select_order_action()
        if self.task=='cmd':
            action, state, values= self.select_cmd_action()
            #print(action,state,values)
        if self.task=='live':
            action, state, values = self.select_live_action()
        return action, state, values

    def select_replay_action(self):
        if self.intention == '告知':
            self.state, self.values = self.state_tacker.updata_state(self.slot)
            if self.state[0]!=1 and self.state[1]!=1:
                self.action='channelName'
            elif self.state[0]!=1 and self.state[1]==1:
                self.action='startDate'
            elif self.state[0]==1 and self.state[1]!=1:
                self.action='startDate'
            elif self.state[0]==1 and self.state[1]==1:
                self.action='more'
        elif self.intention=='nothing':
            self.action = 'done'
            self.state = self.state_tacker.state
            self.values = self.state_tacker.current
        return self.action,self.state,self.values

    def select_order_action(self):
        if self.intention == 'nothing':
            self.action = 'done'
            self.state = self.state_tacker.state
            self.values = self.state_tacker.current
        elif self.intention=='告知':
            self.state, self.values = self.state_tacker.updata_state(self.slot)
            self.action='more'
        return self.action, self.state, self.values

    def select_cmd_action(self):
        self.action=self.slot['cmdValue']
        print(self.action)
        self.state=self.state_tacker.state
        self.values=self.state_tacker.current
        return self.action, self.state, self.values

    def select_live_action(self):
        if self.intention == '告知':
            self.state, self.values = self.state_tacker.updata_state(self.slot)
            if self.state[0] != 1:
                self.action='channelName'
            else:
                self.action='done'
        if self.intention=='nothing':
            self.action = 'done'
            self.state = self.state_tacker.state
            self.values = self.state_tacker.current
        return self.action, self.state, self.values
    '''def choice_action(self,task,intention,state_tacker,slot):
        if task=='huikan':
            if intention == '告知':
                state, values = state_tacker.updata_state(slot)
                action=self.select_action(intention,state)
                print(action)
            elif intention=='noting':
                action='done'
                state=state_tacker.state
                values=state_tacker.current
                print(action)
            return state,values,action'''
    '''def select_huikanaction(self,state_tacker,slot):
        if states[0]!=1 and states[1]!=1:
            action='channelName'
        elif states[0]!=1 and states[1]==1:
            action='startDate'
        elif states[0]==1 and states[1]!=1:
            action='startDate'
        elif states[0]==1 and states[1]==1:
            action='你还要什么问题呢'
        return action'''