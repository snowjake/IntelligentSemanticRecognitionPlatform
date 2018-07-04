from agent.agent import Agent
import numpy as np
import logging


class Task(object):

    def __init__(self, config=None):
        logging.info('fu')
        self.config = config
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


class MultiTask(Task):
    def __init__(self, config):
        logging.info('zi')
        super(MultiTask, self).__init__(config)
        self.agent = Agent(config=config)

    def format_slots(self, output):
        slots = output['slots']
        slots_values = output['slots']
        slots_values_dict = {}
        state = np.zeros(shape=self.config['state_shape'])
        slots_to_index = self.config['slots_to_index']
        for slot, slot_value in zip(slots, slots_values):
            state[slots_to_index[slot]] = 1
            slots_values_dict[slot] = slot_value
        return state

    def step(self, state):
        action = 101
        terminate = False
        # if state in self.terminal_state:
        #     terminate = True
        # else:
        action = self.agent.egreedy_action(state)
        return action, terminate


class SingleTask(Task):
    def format_slots(self, output):
        slots = output['slots']
        slots_values = output['slots']
        slots_values_dict = {}
        state = np.zeros(shape=self.config['state_shape'])
        slots_to_index = self.config['slots_to_index']
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
