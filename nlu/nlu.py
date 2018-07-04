class NLU(object):
    def __init__(self):
        pass

    def raw_to_slots(self, raw_text):
        return {
        'task_id': 1,
        'slots': [],
        'slots_values': [],
        'intent': 'inform'
        }