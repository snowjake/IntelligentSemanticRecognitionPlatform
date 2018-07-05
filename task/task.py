from agent.agent import Agent
import numpy as np
import logging
from functools import wraps


class Task(object):

    def __init__(self, config=None):
        self.config = config
        self.task_id = config['task_id']
        self.terminal_state = config['terminal_state']
        self.state = [0 for _ in range(config['state_shape'])]

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


def parse_output(func):
    @wraps(func)
    def deal_with_special(self, output):
        slots = output['slots']
        slots_values = output['slots_values']
        return func(self, output)
    return deal_with_special


class MultiTask(Task):
    def __init__(self, config):
        super(MultiTask, self).__init__(config)
        self.agent = Agent(config=config)

    '''
    def decorator_name(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not can_run:
            return "Function will not run"
        return f(*args, **kwargs)
    return decorated

    @decorator_name
    def func():
        return("Function is running")
    '''



    @parse_output
    def format_slots(self, output):
        slots = output['slots']
        slots_values = output['slots']
        slots_values_dict = {}
        slots_to_index = self.config['slots_to_index']
        for slot, slot_value in zip(slots, slots_values):
            self.state[slots_to_index[slot]] = 1
            slots_values_dict[slot] = slot_value
        return self.state

    def step(self, state):
        action = 101
        terminate = False
        if state in self.terminal_state:
            terminate = True
        else:
            action = self.agent.egreedy_action(state)
            logging.info('state:{}; action:{}'.format(state, action))
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
