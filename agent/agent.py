from keras.models import Sequential
from keras.layers import Dense
class Agent(object):
    def __init__(self, config):
        # 状态空间的维度
        self.state_dim = config['state_dim']
        # 动作空间的维度
        self.action_dim = config['action_dim']
        # 建立神经网络
        self.wights=r'IntelligentSemanticRecognitionPlatform\dialog\replay.h5'
        self.model = self.model(self.wights, self.action_dim)
    def model(self, weight, naction):
        model = Sequential()
        model.add(Dense(24, input_dim=naction, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(naction, activation='linear'))
        model.load_weights(weight)
        return model
    def egreedy_action(self, state):
        #if self.judge_end(state):
        action = self.model.predict(np.array([state])).argmax()
        return action
        '''else:
            return 'done' '''