class Config(object):
    configs = {
        1: {
            'terminal_state': [[1]],
            'task_id': 1,
            'state_shape': 1,
            'slots_to_index': {'channel_name': 0}
        },
        2: {
            'terminal_state': [[1, 1, 1, 1]],
            'task_id': 2,
            'state_dim': 4,
            'action_dim': 4,
            'state_shape': 4,
            'slots_to_index': {'channel_name': 0}
        },
        3: {
            'terminal_state': [[1, 1, 1, 1]],
            'task_id': 3,
            'state_dim': 4,
            'action_dim': 4,
            'state_shape': 4,
            'slots_to_index': {'channel_name': 0}
        },
        4: {
            'terminal_state': [],
            'task_id': 4,
            'state_dim': 4,
            'action_dim': 4,
            'slots_to_index': {'channel_name': 0}
        }
    }


if __name__ == '__main__':
    print(Config.configs[1])