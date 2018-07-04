import random
import copy


class DialogEnv(object):
    def __init__(self):
        self.step = 0
        self.max_step = 20
        self.state_space = [[0, 0, 0, 0], [1, 0, 0, 0],
                            [0, 0, 0, 1], [1, 0, 0, 1],
                            [0, 0, 1, 0], [1, 0, 1, 0],
                            [0, 0, 1, 1], [1, 0, 1, 1],
                            [0, 1, 0, 0], [1, 1, 0, 0],
                            [0, 1, 0, 1], [1, 1, 0, 1],
                            [0, 1, 1, 0], [1, 1, 1, 0],
                            [0, 1, 1, 1]]

    def reset(self):
        self.step = 0
        return self.state_space[random.randint(0, 14)]

    def interact(self, action, state):
        if action == 0:
            state[0] = 1
        elif action == 1:
            state[1] = 1
        elif action == 2:
            state[2] = 1
        else:
            state[3] = 1

    def gostep(self, action, state):
        state_ = copy.deepcopy(state)
        self.step += 1
        self.interact(action, state_)
        if self.step > self.max_step:
            print('fault')
            done = True
            reward = -1
        elif state_ == [1, 1, 1, 0] or state_ == [1, 1, 1, 1]:
            done = True
            print('success')
            reward = 1
        else:
            done = False
            reward = -0.01
        return state_, reward, done


if __name__ == '__main__':
    for i in range(20):
        print(random.randint(0, 14))