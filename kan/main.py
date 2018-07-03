from task.task import SingleTask, MultiTask
import numpy as np
from config import Config


class TaskManagement(object):

    def __init__(self, max_time_step):
        # state 表示当前对话状态，状态分两种  0:尚未开始 1:进行中
        self.state = 0
        self.time_step = 0
        self.max_time_step = max_time_step
        self.task = None

    def nlg(self, action, terminal):
        '''
        根据action和terminal选择相应的模板
        :param action:
        :param terminal:
        :return:
        '''
        if terminal:
            return '对话结束'
        else:
            return 'action is {}, {}'.format(action, terminal)

    def check(self, output):
        '''
        判断槽值是否合法
        :param output:
        :return:
        '''
        slots = output['slots']
        slots_values = output['slots_values']
        errors = []
        for slot, slots_value in zip(slots, slots_values):
            # 这里对槽值的合法性进行判断
            # 并且记录错误信息
            pass
        return True, errors

    def interact(self, output):
        # is_valid, errors = self.check(output=output)
        # if not is_valid:
        #     return str(errors)
        if output['task_id'] == 1:
            # 如果是任务1，且当前无其他任务，创建任务1
            if self.state == 0:
                config = Config.configs[output['task_id']]
                self.task = SingleTask(config)
            action, terminal = self.step(output=output)
            print(action, terminal)
            return self.task.nlg(action, terminal)
        elif output['task_id'] == 2:
            # 如果是任务2，且当前无其他任务，创建任务2
            if self.state == 0:
                config = Config.configs[output['task_id']]
                self.task = MultiTask(config)
            action, terminal = self.step(output=output)
            return self.nlg(action, terminal)
        elif output['task_id'] == 3:
            # 如果是任务3，且当前无其他任务，创建任务3
            if self.state == 0:
                config = Config.configs[output['task_id']]
                self.task = MultiTask(config)
            action, terminal = self.step(output=output)
            return self.nlg(action, terminal)
        elif output['task_id'] == 4:
            # 如果是任务4，且当前无其他任务，创建任务4
            if self.state == 0:
                config = Config.configs[output['task_id']]
                self.task = MultiTask(config)
            action, terminal = self.step(output=output)
            return self.nlg(action, terminal)
        elif output['task_id'] == 5:
            # 如果是任务5，且当前无其他任务，那么直接转发指令
            if self.state == 0:
                pass
            # 当前有其他任务，那么继承当前任务
            else:
                action, terminal = self.step(output=output)
                return self.nlg(action, terminal)

    def step(self, output):
        '''
        :param output: nlu output
        :return: action
        '''
        terminate = False
        self.time_step += 1
        if self.time_step > self.max_time_step:
            terminate = True
            return 101, terminate
        return self.task.step(self.task.format_slots(output=output))


if __name__ == '__main__':
    output = {
        'task_id': 1,
        'slots': [],
        'slots_values': [],
        'intent': 'inform'
        }
    tm = TaskManagement(10)
    print(tm.interact(output))

