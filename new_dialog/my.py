from flask import Flask
from flask import request
import logging
import sys
from Dialog_Management.DM import *
app = Flask(__name__)
glodal_task1=''
glodal_dialog=None
#nlu=NLU()
#nlg=NLG()
@app.route('/dialog')
def dialog():
    device_id = request.args.get('device_id')
    rawText = request.args.get('rawText')
    my_slot =nlu.rwa_to_slots(rawText)#这个是小广给我的，我传进去的是小广给我的和我之前的
    TreadDates = DM(my_slot).toTreadDate()
    return TreadDates
if __name__ == '__main__':
    print('sss')
    logging.basicConfig(level=logging.INFO,
                    format='levelname:%(levelname)s filename: %(filename)s '
                           'outputNumber: [%(lineno)d]  thread: %(threadName)s output msg:  %(message)s'
                           ' - %(asctime)s', datefmt='[%d/%b/%Y %H:%M:%S]',
                        stream=sys.stdout)

    app.run(host = '0.0.0.0', port = 5000)