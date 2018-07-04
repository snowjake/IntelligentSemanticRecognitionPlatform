class Config(object):
    configs = {
        1: {
            'terminal_state': [[1]],
            'task_id': 1,
            'state_dim': 1,
            'action_dim': 1
        },
        2: {
            'terminal_state': [],
            'task_id': 2,
            'state_dim': 4,
            'action_dim': 4
        },
        3: {
            'terminal_state': [],
            'task_id': 3,
            'state_dim': 4,
            'action_dim': 4
        },
        4: {
            'terminal_state': [],
            'task_id': 4,
            'state_dim': 4,
            'action_dim': 4
        }
    }


if __name__ == '__main__':
    print(Config.configs[1])