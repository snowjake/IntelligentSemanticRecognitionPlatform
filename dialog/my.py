from flask import Flask
from flask import request
import logging
import sys
from Dialog import *
from  nlg import *
from nlu_test.atis_entity_recognition.main import NLU
from control import *
app = Flask(__name__)
glodal_task1=''
glodal_dialog=None
nlu=NLU()
nlg=NLG()
@app.route('/dialog')
def dialog():
    device_id = request.args.get('device_id')
    rawText = request.args.get('rawText')
    my_slot =nlu.rwa_to_slots(rawText)
    print(my_slot)
    global glodal_task1
    global glodal_dialog
    to_nlg, glodal_task1, glodal_dialog =contronl(my_slot,glodal_task1,glodal_dialog )
    respose = nlg.to_speaker(my_slot['task'], to_nlg)
    if to_nlg['action'] == 'done':
        del glodal_dialog
        glodal_dialog = None
        glodal_task1 = ''
    '''if my_slot['task'] == 'cmd':
        cmddia = Dialog(my_slot['task'])
        if my_slot['task']['cmdValue'] in ['PREVIOUSPAGE', 'NEXTPAGE']:
            action = my_slot['task']['cmdValue']
        else:
            action = 'done'
        state, values = cmddia.DM.which_intention(my_slot['intention'], my_slot['slot'])
        #return ({'state': state, 'slots': values, 'action': action})
    elif my_slot['task'] != glodal_task1:
        glodal_task1 = my_slot['task']
        glodal_dialog = Dialog(my_slot['task'])
    #print(glodal_dialog.DM.state_tacker.state)
    print(my_slot['task'],my_slot['intention'], my_slot['slot'])
    state, values = glodal_dialog.DM.which_intention(my_slot['intention'], my_slot['slot'])
    action = glodal_dialog.DM.select_action(state)
    print(glodal_dialog.DM.state_tacker.state,glodal_dialog.DM.state_tacker.current,action)
    to_nlg={'state': state, 'slots': values, 'action': action}'''

    '''task_management = task_dict[device_id]
    reply, terminate = task_management.manage(rawText)
    if terminate:
        task_dict.pop(device_id)'''
    return respose
if __name__ == '__main__':
    print('sss')
    logging.basicConfig(level=logging.INFO,
                    format='levelname:%(levelname)s filename: %(filename)s '
                           'outputNumber: [%(lineno)d]  thread: %(threadName)s output msg:  %(message)s'
                           ' - %(asctime)s', datefmt='[%d/%b/%Y %H:%M:%S]',
                        stream=sys.stdout)

    app.run(host = '0.0.0.0', port = 5000)