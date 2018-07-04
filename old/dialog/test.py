from Dialog import *
glodal_task1=''
glodal_dialog=None
def test(task,intention=None,slot=None):
    global glodal_task1
    global glodal_dialog
    if task=='cmd':
        cmddia=Dialog(task)
        action='done'
        state,values=cmddia.DM.which_intention(intention,slot)
        return ({'state': state, 'slots': values, 'action': action})
    elif task!=glodal_task1:
        glodal_task1=task
        glodal_dialog=Dialog(task)
    state,values=glodal_dialog.DM.which_intention(intention,slot)
    action=glodal_dialog.DM.select_action(state)
    return ({'state':state,'slots':values,'action':action})
x=test('cmd',intention='cmd',slot={'cmdValue':'xx','cmdParam':'1s'})
print(x)