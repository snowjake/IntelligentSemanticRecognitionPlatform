from Dialog import *
def contronl(my_slot,glodal_task1,glodal_dialog):
    if my_slot['task'] == 'cmd':
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
    to_nlg={'state': state, 'slots': values, 'action': action}
    #respose=nlg.to_speaker( my_slot['task'],to_nlg)
    return to_nlg,glodal_task1,glodal_dialog