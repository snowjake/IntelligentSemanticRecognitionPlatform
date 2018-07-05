class NLU(object):
    def __init__(self):
        pass

    def raw_to_slots(self, raw_text):

        return {
        'task_id': 2,
        'slots': [raw_text],
        'slots_values': ['浙江卫视'],
        'intent': 'inform'
        }