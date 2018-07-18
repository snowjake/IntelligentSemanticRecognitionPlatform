class state_tacker:
    def __init__(self, dic):
        self.state = [0] * len(dic)
        #self.history = {}
        self.current = {}
        self.dic = dic
    def updata_state(self, slot):
        for i in slot.items():
            self.state[self.dic[i[0]]] = 1
            self.current[i[0]] = i[1]
            # print(self.state,self.current)
        return self.state, self.current