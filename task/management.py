from task.task import SingleTask, MultiTask
from task.config import Config


class TaskManagement(object):
    def __init__(self, config=None):
        self.nlu = config['nlu']
        self.nlg = config['nlg']
        self.task = None

    def manage(self, raw_text):
        # 首先nlu解析原始语料
        nlu_outputs = self.nlu.raw_to_slots(raw_text=raw_text)
        # 检测到任务1
        if nlu_outputs['task_id'] == 1:
            # 如果当前无其他任务，或者在执行其他任务
            # 立即终止上一个任务，开始任务1
            if self.task == None or self.task.task_id != 1:
                self.task = SingleTask(Config.configs[1])
            action, terminate = self.task.step(self.task.format_slots(nlu_outputs))
            return self.nlg.generate_sentence(action=action, terminate=terminate, task_id=self.task.task_id), terminate
        # 检测到任务2
        elif nlu_outputs['task_id'] == 2:
            # 如果当前无其他任务，或者在执行其他任务
            # 立即终止上一个任务，开始任务2
            if self.task == None or self.task.task_id != 2:
                self.task = MultiTask(Config.configs[2])
            action, terminate = self.task.step(self.task.format_slots(nlu_outputs))
            return self.nlg.generate_sentence(action=action, terminate=terminate, task_id=self.task.task_id), terminate
        elif nlu_outputs['task_id'] == 3:
            # 如果当前无其他任务，或者在执行其他任务
            # 立即终止上一个任务，开始任务3
            if self.task == None or self.task.task_id != 3:
                self.task = MultiTask(Config.configs[3])
            action, terminate = self.task.step(self.task.format_slots(nlu_outputs))
            return self.nlg.generate_sentence(action=action, terminate=terminate, task_id=self.task.task_id), terminate
        elif nlu_outputs['task_id'] == 4:

            pass

