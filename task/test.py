from nlg.nlg import NLG
from nlu.nlu import NLU
from task.management import TaskManagement
if __name__ == '__main__':
    config = {
        'nlu': NLU(),
        'nlg': NLG(),
        # 'task_id': 1,
        'terminal_state': [1]
    }
    tm = TaskManagement(config=config)
    print(tm.manage(''))