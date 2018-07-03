from train.agent.agent import Agent
from train.env.dialog_env import DialogEnv


class DialogManagement(object):

    def __init__(self):
        self.agent = Agent()

    def train(self):
        print('start to train')
        for i in range(0, 10000):
            print('epsilon: ', i)
            env = DialogEnv()
            s = env.reset()
            print('s: ', s)
            while True:
                action = self.agent.egreedy_action(s)
                s_, reward, done = env.gostep(action, s)
                print(action)
                print('s_:', s_)
                self.agent.precive(s, action, reward, s_, done)
                s = s_
                if done:
                    break
                self.agent.save_model(step=i)


if __name__ == '__main__':
    dm = DialogManagement()
    dm.train()
