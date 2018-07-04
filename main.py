from flask import Flask
from flask import request
import logging
import sys
from nlu.nlu import NLU
from nlg.nlg import NLG
from task.management import TaskManagement
app = Flask(__name__)


@app.route('/dialog')
def dialog():
    device_id = request.args.get('device_id')
    rawText = request.args.get('rawText')
    if device_id not in task_dict:
        config = {
            'nlu': NLU(),
            'nlg': NLG(),
            'terminal_state': [1]
        }
        task_dict[device_id] = TaskManagement(config=config)
    task_management = task_dict[device_id]
    reply, terminate = task_management.manage(rawText)
    if terminate:
        task_dict.pop(device_id)
    return reply

@app.route('/')
def hello_world():
    return 'ddddd'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='levelname:%(levelname)s filename: %(filename)s '
                           'outputNumber: [%(lineno)d]  thread: %(threadName)s output msg:  %(message)s'
                           ' - %(asctime)s', datefmt='[%d/%b/%Y %H:%M:%S]',
                        stream=sys.stdout)
    nlu = NLU()
    nlg = NLG()
    task_dict = {}
    app.run(debug=True)