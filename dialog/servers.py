import socket
import json
from Dialog import *
HOST='localhost'
PORT=50007
s= socket.socket(socket.AF_INET,socket.SOCK_STREAM)   #定义socket类型，网络通信，TCP
s.bind((HOST,PORT))   #套接字绑定的IP与端口
s.listen(1)
glodal_task1=''
glodal_dialog=None
def test(task,intention,slot):
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
flag=1#开始TCP监听
while flag:
       conn,addr=s.accept()   #接受TCP连接，并返回新的套接字与IP地址
       print ('Connected by',addr)  #输出客户端的IP地址
       while 1:
                data=conn.recv(1024)    #把接收的数据实例化
                #print(data.decode())  #commands.getstatusoutput执行系统命令（即shell命令），返回两个结果，第一个是状态，成功则为0，第二个是执行成功或失败的输出信息
                josn_dict=json.loads(data.decode())
                aaa=test(josn_dict['task'], josn_dict['intention'], josn_dict['slot'])
                print(aaa)
                if aaa['action']=='done':
                        conn.send('Done.'.encode())
                        print('完成')
                        #flag=0
                        del glodal_dialog,glodal_task1
                        glodal_dialog=None
                        glodal_task1=''

                else:
                       conn.send(aaa['action'].encode())
