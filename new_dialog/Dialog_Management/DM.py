from Dialog_Management.state_tacker import *
from Dialog_Management.Dictionry import *
from Dialog_Management.action_select import *
class DM:
    def __init__(self,TreadDate):
        self.TreadDate=TreadDate
        self.nlu_result=self.TreadDate['nlu_result']
        self.DM_result=self.TreadDate['DM_result']
        if len(self.DM_result)==0:
            self.DM_result['state']=[]
            self.DM_result['current']=[]
        self.task = self.nlu_result['task']
        self.dictory = Dictionry(self.task)
        self.state_tacker = state_tacker(self.dictory.dictory)
        self.intention=self.nlu_result['intention']
        self.slot=self.nlu_result['slot']
        self.action_s = action_select(self.task, self.intention, self.state_tacker, self.slot)
        if self.DM_result['state'] != []:
            self.state_tacker.state = self.DM_result['state']
            self.state_tacker.current = self.DM_result['current']
        self.action,self.state_tacker.state, self.state_tacker.current= \
            self.action_s.choice_action()
        '''if self.task == 'huikan':
            self.dictory = Dictionry(self.task)
            self.state_tacker = state_tacker(self.dictory.dictory)
            if DM_result['state']!=[]:
                self.state_tacker.state=DM_result['state']
                self.state_tacker.current=DM_result['current']
        if self.task == 'dianbo':
            self.dictory = Dictionry(self.task)
            self.state_tacker = state_tacker(self.dictory.dictory)
            if DM_result['state']!=[]:
                self.state_tacker.state=DM_result['state']
                self.state_tacker.current=DM_result['current']
        if self.task == 'live':
            self.dictory = Dictionry(self.task)
            self.state_tacker = state_tacker(self.dictory.dictory)
            if DM_result['state']!=[]:
                self.state_tacker.state=DM_result['state']
                self.state_tacker.current=DM_result['current']
        if self.task=='cmd':
            self.dictory = Dictionry(self.task)
            self.state_tacker = state_tacker(self.dictory.dictory)'''
    def call_api(self, task):
        pass
    def toTreadDate(self):
        self.DM_result['state']=self.state_tacker.state
        self.DM_result['current']=self.state_tacker.current
        self.DM_result['action']=self.action
        #if self.task=='cmd' or self.action=='done':
            #self.list.pop(0)
        return self.TreadDate
    '''def which_intention(self):
        if self.intention == '告知':
            self.state, self.values = self.state_tacker.updata_state(self.slot)
            #return state, values
        if self.intention=='cmd':
            self.state, self.values = self.state_tacker.updata_state(self.slot)
        if self.intention=='nothing':
            #return state, values
    def select_action(self):

        if self.judge_end(self.state):
            #action = self.model.predict(np.array([state])).argmax()
            action = self.model.loc[str(state)].idxmax()
            return (self.dictory.action[int(action)])
        else:
            return 'done' '''
'''nlu_result={'task':'huikan','intention':'告知','slot':{'channelName':'xxxx'}}
DM_result={'state':[0,1,1,1,1],'current':{
                                          'startDate':'xx', 'startTime':'xxx', 'name':'xxx'
    ,'endTime':'xxxxxx'}}'''
'''nlu_result={'task':'dianbo','intention':'nothing','slot':{'category':'喜剧'}}
DM_result={'state':[1,0,0,1,0,0,1,0,1],'current':{'videoName':'三生三世十里桃花','area':'中国',
                         'episode':'第一集','endyear':'2016'}}'''
TreadDate={'nlu_result':{'task':'huikan','intention':'nothing','slot':{'channelName':'xxxx',
                                                                  'startTime':'2018-1-10'}},
'DM_result':{'state':[0,0,0,0,0],'current':{}}}
x=[TreadDate,1]
TreadDates=DM(x[0]).toTreadDate()
x[0]=TreadDates
print(x[0])