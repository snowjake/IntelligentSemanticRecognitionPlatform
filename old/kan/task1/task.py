from agent.agent import Agent
import numpy as np


class Task(object):

    def __init__(self, config=None):
        self.task_id = config['task_id']
        self.terminal_state = config['terminal_state']

    def format_slots(self, output):
        '''
        将nlu输出结构化
        :param task_id:
        :param slots:
        :param slots_values:
        :return: state
        '''
        pass

    def step(self, state):
        pass

    def nlg(self, action):
        pass


class MultiTask(Task):
    def __init__(self, config=None):
        self.agent = Agent(config=config)

    def step(self, state):
        action = 101
        terminate = False
        if state in self.terminal_state:
            terminate = True
        else:
            action = self.agent.egreedy_action(state)
        return action, terminate

    def nlg(self, action):
        return 'action is: {}'.format(action)


class SingleTask(Task):
    def format_slots(self, output):
        slots = output['slots']
        slots_values = output['slots']
        slots_values_dict = {}
        state = np.zeros(shape=1)
        slots_to_index = {
            'channel_name': 0}
        for slot, slot_value in zip(slots, slots_values):
            state[slots_to_index[slot]] = 1
            slots_values_dict[slot] = slot_value
        return state

    def step(self, state):
        terminate = False
        action = 101
        if state in self.terminal_state:
            terminate = True
        else:
            action = 0
        return action, terminate

    def nlg(self, action, terminal):
        if action == 0:
            return '请问您要看哪个频道？'


if __name__ == '__main__':
    pass